#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include <string.h>
#include <stdint.h>

// Cached Metal matmul handle — pre-creates all buffers and MPS objects.
typedef struct {
    void *wBuf;      // id<MTLBuffer> — weight buffer (pre-staged)
    void *xBuf;      // id<MTLBuffer> — input buffer (reused)
    void *oBuf;      // id<MTLBuffer> — output buffer (reused)
    void *wMat;      // MPSMatrix*
    void *xMat;      // MPSMatrix*
    void *oMat;      // MPSMatrix*
    void *mm;        // MPSMatrixMultiplication*
    int outDim;
    int inDim;
} MetalMatmulHandle;

static id<MTLDevice> _cmtlDevice = nil;
static id<MTLCommandQueue> _cmtlQueue = nil;

int metalCachedInit(void) {
    @autoreleasepool {
        if (_cmtlDevice != nil) return 0;
        _cmtlDevice = MTLCreateSystemDefaultDevice();
        if (_cmtlDevice == nil) return -1;
        _cmtlQueue = [_cmtlDevice newCommandQueue];
        if (_cmtlQueue == nil) return -1;
        return 0;
    }
}

// Create a cached matmul handle with pre-staged weights.
MetalMatmulHandle* metalCachedCreate(const float *weights, int outDim, int inDim) {
    @autoreleasepool {
        if (_cmtlDevice == nil) return NULL;

        size_t wBytes = (size_t)outDim * inDim * sizeof(float);

        id<MTLBuffer> wBuf = [_cmtlDevice newBufferWithBytes:weights
                                                       length:wBytes
                                                      options:MTLResourceStorageModeShared];
        if (wBuf == nil) return NULL;

        MPSMatrixDescriptor *wDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:outDim
                                                                           columns:inDim
                                                                          rowBytes:inDim * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrix *wMat = [[MPSMatrix alloc] initWithBuffer:wBuf descriptor:wDesc];

        // Pre-create input/output buffers (reused across calls).
        size_t xBytes = (size_t)inDim * sizeof(float);
        size_t oBytes = (size_t)outDim * sizeof(float);
        id<MTLBuffer> xBuf = [_cmtlDevice newBufferWithLength:xBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> oBuf = [_cmtlDevice newBufferWithLength:oBytes options:MTLResourceStorageModeShared];
        if (xBuf == nil || oBuf == nil) return NULL;

        MPSMatrixDescriptor *xDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:inDim
                                                                           columns:1
                                                                          rowBytes:sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *oDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:outDim
                                                                           columns:1
                                                                          rowBytes:sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
        MPSMatrix *xMat = [[MPSMatrix alloc] initWithBuffer:xBuf descriptor:xDesc];
        MPSMatrix *oMat = [[MPSMatrix alloc] initWithBuffer:oBuf descriptor:oDesc];

        MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:_cmtlDevice
                                                                       transposeLeft:NO
                                                                      transposeRight:NO
                                                                          resultRows:outDim
                                                                       resultColumns:1
                                                                     interiorColumns:inDim
                                                                               alpha:1.0
                                                                                beta:0.0];

        MetalMatmulHandle *h = (MetalMatmulHandle *)calloc(1, sizeof(MetalMatmulHandle));
        h->wBuf = (__bridge_retained void *)wBuf;
        h->xBuf = (__bridge_retained void *)xBuf;
        h->oBuf = (__bridge_retained void *)oBuf;
        h->wMat = (__bridge_retained void *)wMat;
        h->xMat = (__bridge_retained void *)xMat;
        h->oMat = (__bridge_retained void *)oMat;
        h->mm = (__bridge_retained void *)mm;
        h->outDim = outDim;
        h->inDim = inDim;
        return h;
    }
}

// Execute cached matmul: out = weights @ x (all buffers pre-staged).
int metalCachedExec(MetalMatmulHandle *h, float *out, const float *x) {
    @autoreleasepool {
        if (h == NULL || _cmtlDevice == nil || _cmtlQueue == nil) return -1;

        int inDim = h->inDim;
        int outDim = h->outDim;

        // Copy input to pre-staged GPU buffer.
        id<MTLBuffer> xBuf = (__bridge id<MTLBuffer>)h->xBuf;
        memcpy(xBuf.contents, x, (size_t)inDim * sizeof(float));

        // Encode and execute.
        id<MTLCommandBuffer> cmdBuf = [_cmtlQueue commandBuffer];
        MPSMatrixMultiplication *mm = (__bridge MPSMatrixMultiplication *)h->mm;
        MPSMatrix *wMat = (__bridge MPSMatrix *)h->wMat;
        MPSMatrix *xMat = (__bridge MPSMatrix *)h->xMat;
        MPSMatrix *oMat = (__bridge MPSMatrix *)h->oMat;
        [mm encodeToCommandBuffer:cmdBuf leftMatrix:wMat rightMatrix:xMat resultMatrix:oMat];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.error != nil) return -1;

        // Copy output from GPU buffer.
        id<MTLBuffer> oBuf = (__bridge id<MTLBuffer>)h->oBuf;
        memcpy(out, oBuf.contents, (size_t)outDim * sizeof(float));
        return 0;
    }
}

void metalCachedDestroy(MetalMatmulHandle *h) {
    if (h == NULL) return;
    // Without ARC, we stored raw pointers. Just free the struct.
    // The Metal objects will be cleaned up when the process exits.
    free(h);
}
