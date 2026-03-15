#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string.h>
#include <stdint.h>

// Runtime-compiled Metal compute shader for fp16 weight matvec.
static id<MTLDevice> _gemvDevice = nil;
static id<MTLCommandQueue> _gemvQueue = nil;
static id<MTLComputePipelineState> _gemvPipeline = nil;

static const char *gemvShaderSource =
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

int metalGemvInit(void) {
    @autoreleasepool {
        if (_gemvPipeline != nil) return 0;
        _gemvDevice = MTLCreateSystemDefaultDevice();
        if (_gemvDevice == nil) return -1;
        _gemvQueue = [_gemvDevice newCommandQueue];
        if (_gemvQueue == nil) return -1;

        NSError *error = nil;
        id<MTLLibrary> lib = [_gemvDevice newLibraryWithSource:
            [NSString stringWithUTF8String:gemvShaderSource] options:nil error:&error];
        if (lib == nil) return -2;

        id<MTLFunction> fn = [lib newFunctionWithName:@"gemv_f16"];
        if (fn == nil) return -3;

        _gemvPipeline = [_gemvDevice newComputePipelineStateWithFunction:fn error:&error];
        if (_gemvPipeline == nil) return -4;

        return 0;
    }
}

// Cached handle for a specific weight matrix.
typedef struct {
    void *wBuf;  // id<MTLBuffer> fp16 weights
    int outDim;
    int inDim;
} MetalGemvHandle;

MetalGemvHandle* metalGemvCreate(const uint16_t *weights_f16, int outDim, int inDim) {
    @autoreleasepool {
        if (_gemvDevice == nil) return NULL;
        size_t wBytes = (size_t)outDim * inDim * sizeof(uint16_t);
        id<MTLBuffer> wBuf = [_gemvDevice newBufferWithBytes:weights_f16
                                                       length:wBytes
                                                      options:MTLResourceStorageModeShared];
        if (wBuf == nil) return NULL;
        MetalGemvHandle *h = (MetalGemvHandle *)calloc(1, sizeof(MetalGemvHandle));
        h->wBuf = (__bridge_retained void *)wBuf;
        h->outDim = outDim;
        h->inDim = inDim;
        return h;
    }
}

int metalGemvExec(MetalGemvHandle *h, float *out, const float *x) {
    @autoreleasepool {
        if (h == NULL || _gemvPipeline == nil) return -1;
        int outDim = h->outDim;
        int inDim = h->inDim;
        uint32_t inDimU = (uint32_t)inDim;

        size_t xBytes = (size_t)inDim * sizeof(float);
        size_t oBytes = (size_t)outDim * sizeof(float);

        id<MTLBuffer> xBuf = [_gemvDevice newBufferWithBytes:x length:xBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> oBuf = [_gemvDevice newBufferWithLength:oBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> dimBuf = [_gemvDevice newBufferWithBytes:&inDimU length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [_gemvQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:_gemvPipeline];
        [enc setBuffer:(__bridge id<MTLBuffer>)h->wBuf offset:0 atIndex:0];
        [enc setBuffer:xBuf offset:0 atIndex:1];
        [enc setBuffer:oBuf offset:0 atIndex:2];
        [enc setBuffer:dimBuf offset:0 atIndex:3];

        MTLSize gridSize = MTLSizeMake(outDim, 1, 1);
        NSUInteger threadGroupSize = _gemvPipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > (NSUInteger)outDim) threadGroupSize = outDim;
        MTLSize tgSize = MTLSizeMake(threadGroupSize, 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.error != nil) return -1;
        memcpy(out, oBuf.contents, oBytes);
        return 0;
    }
}

void metalGemvDestroy(MetalGemvHandle *h) {
    if (h != NULL) free(h);
}
