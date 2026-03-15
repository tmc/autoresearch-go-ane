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
