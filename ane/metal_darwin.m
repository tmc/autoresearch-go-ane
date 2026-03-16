#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include <string.h>

static id<MTLDevice> _mtlDevice = nil;
static id<MTLCommandQueue> _mtlQueue = nil;

int metalGEMV_init(int maxOutDim, int maxInDim) {
    @autoreleasepool {
        _mtlDevice = MTLCreateSystemDefaultDevice();
        if (_mtlDevice == nil) return -1;
        _mtlQueue = [_mtlDevice newCommandQueue];
        if (_mtlQueue == nil) return -1;
        return 0;
    }
}

int metalGEMV_exec(float *out, const float *weights, const float *x, int outDim, int inDim) {
    @autoreleasepool {
        if (_mtlDevice == nil || _mtlQueue == nil) return -1;

        size_t wBytes = (size_t)outDim * inDim * sizeof(float);
        size_t xBytes = (size_t)inDim * sizeof(float);
        size_t oBytes = (size_t)outDim * sizeof(float);

        // Create Metal buffers (no-copy where possible for large weight matrices).
        id<MTLBuffer> wBuf = [_mtlDevice newBufferWithBytesNoCopy:(void *)weights
                                                           length:wBytes
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
        if (wBuf == nil) {
            // Fallback: copy if no-copy fails (alignment requirements).
            wBuf = [_mtlDevice newBufferWithBytes:weights
                                           length:wBytes
                                          options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> xBuf = [_mtlDevice newBufferWithBytes:x
                                                     length:xBytes
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> oBuf = [_mtlDevice newBufferWithLength:oBytes
                                                     options:MTLResourceStorageModeShared];

        if (wBuf == nil || xBuf == nil || oBuf == nil) return -1;

        // MPS matrix multiply: out = weights * x
        // weights: [outDim, inDim], x: [inDim, 1], out: [outDim, 1]
        MPSMatrixDescriptor *wDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:outDim
                                                                           columns:inDim
                                                                          rowBytes:inDim * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *xDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:inDim
                                                                           columns:1
                                                                          rowBytes:sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *oDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:outDim
                                                                           columns:1
                                                                          rowBytes:sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];

        MPSMatrix *wMat = [[MPSMatrix alloc] initWithBuffer:wBuf descriptor:wDesc];
        MPSMatrix *xMat = [[MPSMatrix alloc] initWithBuffer:xBuf descriptor:xDesc];
        MPSMatrix *oMat = [[MPSMatrix alloc] initWithBuffer:oBuf descriptor:oDesc];

        MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:_mtlDevice
                                                                       transposeLeft:NO
                                                                      transposeRight:NO
                                                                          resultRows:outDim
                                                                       resultColumns:1
                                                                     interiorColumns:inDim
                                                                               alpha:1.0
                                                                                beta:0.0];

        id<MTLCommandBuffer> cmdBuf = [_mtlQueue commandBuffer];
        [mm encodeToCommandBuffer:cmdBuf leftMatrix:wMat rightMatrix:xMat resultMatrix:oMat];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.error != nil) return -1;

        // Copy result back to caller.
        memcpy(out, oBuf.contents, oBytes);
        return 0;
    }
}

void metalGEMV_cleanup(void) {
    _mtlQueue = nil;
    _mtlDevice = nil;
}

// ---- FP16 compute shader GEMV for decode ----

static id<MTLComputePipelineState> _fp16Pipeline = nil;

static const char *fp16GemvShader =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void gemv_f16(\n"
    "    device const half *weights [[buffer(0)]],\n"
    "    device const float *x      [[buffer(1)]],\n"
    "    device float *out          [[buffer(2)]],\n"
    "    constant uint &inDim       [[buffer(3)]],\n"
    "    uint tid [[thread_position_in_grid]]\n"
    ") {\n"
    "    uint row = tid;\n"
    "    device const half *w_row = weights + (uint64_t)row * inDim;\n"
    "    float sum = 0.0f;\n"
    "    for (uint d = 0; d < inDim; d += 4) {\n"
    "        half4 wv = *((device const half4 *)(w_row + d));\n"
    "        float4 xv = *((device const float4 *)(x + d));\n"
    "        sum += dot(float4(wv), xv);\n"
    "    }\n"
    "    out[row] = sum;\n"
    "}\n";

static int ensureFP16Pipeline(void) {
    if (_fp16Pipeline != nil) return 0;
    if (_mtlDevice == nil) return -1;
    @autoreleasepool {
        NSError *error = nil;
        id<MTLLibrary> lib = [_mtlDevice newLibraryWithSource:
            [NSString stringWithUTF8String:fp16GemvShader] options:nil error:&error];
        if (lib == nil) return -2;
        id<MTLFunction> fn = [lib newFunctionWithName:@"gemv_f16"];
        if (fn == nil) return -3;
        _fp16Pipeline = [_mtlDevice newComputePipelineStateWithFunction:fn error:&error];
        if (_fp16Pipeline == nil) return -4;
        return 0;
    }
}

typedef struct {
    void *wBuf;    // id<MTLBuffer> — persistent fp16 weights on GPU
    void *xBuf;    // id<MTLBuffer> — reusable input buffer
    void *oBuf;    // id<MTLBuffer> — reusable output buffer
    void *dimBuf;  // id<MTLBuffer> — cached inDim constant
    int outDim;
    int inDim;
    NSUInteger tgSize; // cached threadgroup size
} MetalFP16Handle;

// Create a cached fp16 weight buffer on GPU with pre-allocated I/O buffers.
MetalFP16Handle* metalFP16GemvCreate(const uint16_t *weights_f16, int outDim, int inDim) {
    @autoreleasepool {
        if (_mtlDevice == nil || ensureFP16Pipeline() != 0) return NULL;
        size_t wBytes = (size_t)outDim * inDim * sizeof(uint16_t);
        size_t xBytes = (size_t)inDim * sizeof(float);
        size_t oBytes = (size_t)outDim * sizeof(float);

        id<MTLBuffer> wBuf = [_mtlDevice newBufferWithBytes:weights_f16
                                                       length:wBytes
                                                      options:MTLResourceStorageModeShared];
        if (wBuf == nil) return NULL;

        // Pre-allocate reusable I/O buffers.
        id<MTLBuffer> xBuf = [_mtlDevice newBufferWithLength:xBytes
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> oBuf = [_mtlDevice newBufferWithLength:oBytes
                                                     options:MTLResourceStorageModeShared];
        uint32_t inDimU = (uint32_t)inDim;
        id<MTLBuffer> dimBuf = [_mtlDevice newBufferWithBytes:&inDimU
                                                       length:sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared];

        if (xBuf == nil || oBuf == nil || dimBuf == nil) return NULL;

        NSUInteger tgSize = _fp16Pipeline.maxTotalThreadsPerThreadgroup;
        if (tgSize > (NSUInteger)outDim) tgSize = outDim;

        MetalFP16Handle *h = (MetalFP16Handle *)calloc(1, sizeof(MetalFP16Handle));
        h->wBuf = (__bridge_retained void *)wBuf;
        h->xBuf = (__bridge_retained void *)xBuf;
        h->oBuf = (__bridge_retained void *)oBuf;
        h->dimBuf = (__bridge_retained void *)dimBuf;
        h->outDim = outDim;
        h->inDim = inDim;
        h->tgSize = tgSize;
        return h;
    }
}

// Execute fp16 GEMV: out = weights_fp16 @ x_fp32
// Uses pre-allocated buffers — only copies input, dispatches, copies output.
int metalFP16GemvExec(MetalFP16Handle *h, float *out, const float *x) {
    if (h == NULL || _fp16Pipeline == nil || _mtlQueue == nil) return -1;
    @autoreleasepool {
        int outDim = h->outDim;
        int inDim = h->inDim;

        // Copy input to pre-allocated GPU buffer.
        id<MTLBuffer> xBuf = (__bridge id<MTLBuffer>)h->xBuf;
        memcpy(xBuf.contents, x, (size_t)inDim * sizeof(float));

        id<MTLCommandBuffer> cmdBuf = [_mtlQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:_fp16Pipeline];
        [enc setBuffer:(__bridge id<MTLBuffer>)h->wBuf offset:0 atIndex:0];
        [enc setBuffer:xBuf offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)h->oBuf offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)h->dimBuf offset:0 atIndex:3];

        [enc dispatchThreads:MTLSizeMake(outDim, 1, 1) threadsPerThreadgroup:MTLSizeMake(h->tgSize, 1, 1)];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.error != nil) return -1;

        // Copy output from GPU buffer.
        id<MTLBuffer> oBuf = (__bridge id<MTLBuffer>)h->oBuf;
        memcpy(out, oBuf.contents, (size_t)outDim * sizeof(float));
        return 0;
    }
}

void metalFP16GemvDestroy(MetalFP16Handle *h) {
    if (h != NULL) free(h);
}
