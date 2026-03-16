#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>
#include <string.h>
#include <stdint.h>

// MPSGraph-based transformer decode: entire model in ONE compiled graph.
// Eliminates per-layer Metal dispatch overhead.

typedef struct {
    void *graph;       // MPSGraph*
    void *executable;  // MPSGraphExecutable* (compiled)
    void *device;      // id<MTLDevice>
    void *cmdQueue;    // id<MTLCommandQueue>
    // Weight tensors (persistent on GPU in fp16)
    void **wq;         // per-layer Wq [dim, qDim] fp16
    void **wk;         // per-layer Wk [dim, kvDim] fp16
    void **wv;         // per-layer Wv [dim, kvDim] fp16
    void **wo;         // per-layer Wo [qDim, dim] fp16
    void **w1;         // per-layer W1 [dim, hidden] fp16
    void **w3;         // per-layer W3 [dim, hidden] fp16
    void **w2;         // per-layer W2 [hidden, dim] fp16
    void **rmsAtt;     // per-layer RMS attention [dim] fp32
    void **rmsFFN;     // per-layer RMS FFN [dim] fp32
    void *rmsFinal;    // final RMS [dim] fp32
    void *embed;       // embedding [vocab, dim] fp16
    int nLayers;
    int dim;
    int qDim;
    int kvDim;
    int hidden;
    int heads;
    int kvHeads;
    int headDim;
    int vocab;
} MPSGraphTransformer;

// Helper: create an MPSGraphTensor from fp16 weight data.
static MPSGraphTensor* constantFP16(MPSGraph *graph, const uint16_t *data, int rows, int cols, NSString *name) {
    NSData *nsdata = [NSData dataWithBytes:data length:(NSUInteger)rows * cols * sizeof(uint16_t)];
    return [graph constantWithData:nsdata shape:@[@(rows), @(cols)] dataType:MPSDataTypeFloat16];
}

// Helper: create an MPSGraphTensor from fp32 weight data.
static MPSGraphTensor* constantFP32(MPSGraph *graph, const float *data, int size, NSString *name) {
    NSData *nsdata = [NSData dataWithBytes:data length:(NSUInteger)size * sizeof(float)];
    return [graph constantWithData:nsdata shape:@[@1, @(size)] dataType:MPSDataTypeFloat32];
}

// RMSNorm as graph operations: out = x * rsqrt(mean(x^2) + eps) * w
static MPSGraphTensor* graphRMSNorm(MPSGraph *graph, MPSGraphTensor *x, MPSGraphTensor *w, int dim, NSString *prefix) {
    // x: [1, dim], w: [1, dim]
    MPSGraphTensor *x2 = [graph multiplicationWithPrimaryTensor:x secondaryTensor:x name:[prefix stringByAppendingString:@"_sq"]];
    MPSGraphTensor *mean = [graph meanOfTensor:x2 axes:@[@1] name:[prefix stringByAppendingString:@"_mean"]];
    MPSGraphTensor *eps = [graph constantWithScalar:1e-5 dataType:MPSDataTypeFloat32];
    MPSGraphTensor *meanEps = [graph additionWithPrimaryTensor:mean secondaryTensor:eps name:[prefix stringByAppendingString:@"_me"]];
    MPSGraphTensor *rsqrt = [graph reciprocalSquareRootWithTensor:meanEps name:[prefix stringByAppendingString:@"_rsqrt"]];
    MPSGraphTensor *normed = [graph multiplicationWithPrimaryTensor:x secondaryTensor:rsqrt name:[prefix stringByAppendingString:@"_norm"]];
    return [graph multiplicationWithPrimaryTensor:normed secondaryTensor:w name:[prefix stringByAppendingString:@"_out"]];
}

// GEMV via matmul: out = x @ W^T where W is [outDim, inDim] in fp16
// x: [1, inDim] fp32, W: [outDim, inDim] fp16, out: [1, outDim] fp32
static MPSGraphTensor* graphGEMV(MPSGraph *graph, MPSGraphTensor *x, MPSGraphTensor *W, NSString *name) {
    // Cast x to fp16 for the matmul
    MPSGraphTensor *x16 = [graph castTensor:x toType:MPSDataTypeFloat16 name:[name stringByAppendingString:@"_xf16"]];
    // W is [outDim, inDim], x16 is [1, inDim]. We want [1, inDim] @ [inDim, outDim] = [1, outDim]
    // So transpose W: [outDim, inDim] -> [inDim, outDim]
    MPSGraphTensor *Wt = [graph transposeTensor:W dimension:0 withDimension:1 name:[name stringByAppendingString:@"_wt"]];
    MPSGraphTensor *y16 = [graph matrixMultiplicationWithPrimaryTensor:x16 secondaryTensor:Wt name:name];
    // Cast back to fp32
    return [graph castTensor:y16 toType:MPSDataTypeFloat32 name:[name stringByAppendingString:@"_f32"]];
}

// SiLU: x * sigmoid(x)
static MPSGraphTensor* graphSiLU(MPSGraph *graph, MPSGraphTensor *x, NSString *name) {
    MPSGraphTensor *sig = [graph sigmoidWithTensor:x name:[name stringByAppendingString:@"_sig"]];
    return [graph multiplicationWithPrimaryTensor:x secondaryTensor:sig name:name];
}

// Build and compile the full transformer decode graph.
// Returns NULL on failure.
MPSGraphTransformer* mpsGraphTransformerCreate(
    int nLayers, int dim, int qDim, int kvDim, int hidden,
    int heads, int kvHeads, int headDim, int vocab,
    const float *rmsFinalWeights,
    // Per-layer weights (arrays of nLayers pointers):
    const uint16_t **wqAll, const uint16_t **wkAll, const uint16_t **wvAll, const uint16_t **woAll,
    const uint16_t **w1All, const uint16_t **w3All, const uint16_t **w2All,
    const float **rmsAttAll, const float **rmsFFNAll,
    const uint16_t *embedWeights,
    float residualScale
) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return NULL;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return NULL;

        MPSGraph *graph = [[MPSGraph alloc] init];

        // Input: x [1, dim] fp32
        MPSGraphTensor *x = [graph placeholderWithShape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:@"input"];

        MPSGraphTensor *cur = x;
        MPSGraphTensor *rsConst = [graph constantWithScalar:residualScale dataType:MPSDataTypeFloat32];

        for (int li = 0; li < nLayers; li++) {
            NSString *lp = [NSString stringWithFormat:@"l%d_", li]; // layer prefix

            // === Attention block ===
            // RMSNorm
            MPSGraphTensor *rmsAttW = constantFP32(graph, rmsAttAll[li], dim, [lp stringByAppendingString:@"rms_att"]);
            MPSGraphTensor *xn = graphRMSNorm(graph, cur, rmsAttW, dim, [lp stringByAppendingString:@"rms1"]);

            // Q, K, V projections
            MPSGraphTensor *Wq = constantFP16(graph, wqAll[li], qDim, dim, [lp stringByAppendingString:@"Wq"]);
            MPSGraphTensor *Wk = constantFP16(graph, wkAll[li], kvDim, dim, [lp stringByAppendingString:@"Wk"]);
            MPSGraphTensor *Wv = constantFP16(graph, wvAll[li], kvDim, dim, [lp stringByAppendingString:@"Wv"]);

            MPSGraphTensor *q = graphGEMV(graph, xn, Wq, [lp stringByAppendingString:@"q"]);
            MPSGraphTensor *k = graphGEMV(graph, xn, Wk, [lp stringByAppendingString:@"k"]);
            MPSGraphTensor *v = graphGEMV(graph, xn, Wv, [lp stringByAppendingString:@"v"]);

            // NOTE: RoPE and attention against KV cache are done on CPU.
            // This graph only handles the matmul portions.
            // For now, output Q, K, V for CPU processing.
            // TODO: Add RoPE and attention to graph for full GPU decode.

            // For the initial version, we'll just do the matmul graph to
            // measure the overhead elimination. The full graph needs
            // RoPE + KV cache attention which requires more work.

            // Wo projection (using Q as placeholder for attention output)
            MPSGraphTensor *Wo = constantFP16(graph, woAll[li], dim, qDim, [lp stringByAppendingString:@"Wo"]);
            MPSGraphTensor *attOut = graphGEMV(graph, q, Wo, [lp stringByAppendingString:@"wo"]); // placeholder

            // Residual
            MPSGraphTensor *diff = [graph subtractionWithPrimaryTensor:attOut secondaryTensor:cur name:[lp stringByAppendingString:@"att_diff"]];
            MPSGraphTensor *scaled = [graph multiplicationWithPrimaryTensor:diff secondaryTensor:rsConst name:[lp stringByAppendingString:@"att_scaled"]];
            MPSGraphTensor *x2 = [graph additionWithPrimaryTensor:cur secondaryTensor:scaled name:[lp stringByAppendingString:@"x2"]];

            // === FFN block ===
            MPSGraphTensor *rmsFFNW = constantFP32(graph, rmsFFNAll[li], dim, [lp stringByAppendingString:@"rms_ffn"]);
            MPSGraphTensor *xn2 = graphRMSNorm(graph, x2, rmsFFNW, dim, [lp stringByAppendingString:@"rms2"]);

            MPSGraphTensor *W1 = constantFP16(graph, w1All[li], hidden, dim, [lp stringByAppendingString:@"W1"]);
            MPSGraphTensor *W3 = constantFP16(graph, w3All[li], hidden, dim, [lp stringByAppendingString:@"W3"]);
            MPSGraphTensor *W2 = constantFP16(graph, w2All[li], dim, hidden, [lp stringByAppendingString:@"W2"]);

            MPSGraphTensor *h1 = graphGEMV(graph, xn2, W1, [lp stringByAppendingString:@"h1"]);
            MPSGraphTensor *h3 = graphGEMV(graph, xn2, W3, [lp stringByAppendingString:@"h3"]);

            // SiLU gate
            MPSGraphTensor *silu = graphSiLU(graph, h1, [lp stringByAppendingString:@"silu"]);
            MPSGraphTensor *gate = [graph multiplicationWithPrimaryTensor:silu secondaryTensor:h3 name:[lp stringByAppendingString:@"gate"]];

            // W2
            MPSGraphTensor *ff = graphGEMV(graph, gate, W2, [lp stringByAppendingString:@"ff"]);

            // Residual
            MPSGraphTensor *ffDiff = [graph subtractionWithPrimaryTensor:ff secondaryTensor:x2 name:[lp stringByAppendingString:@"ff_diff"]];
            MPSGraphTensor *ffScaled = [graph multiplicationWithPrimaryTensor:ffDiff secondaryTensor:rsConst name:[lp stringByAppendingString:@"ff_scaled"]];
            cur = [graph additionWithPrimaryTensor:x2 secondaryTensor:ffScaled name:[lp stringByAppendingString:@"next"]];
        }

        // Final RMSNorm
        MPSGraphTensor *rmsFinalW = constantFP32(graph, rmsFinalWeights, dim, @"rms_final");
        MPSGraphTensor *finalNorm = graphRMSNorm(graph, cur, rmsFinalW, dim, @"final_rms");

        // Classifier
        MPSGraphTensor *embedW = constantFP16(graph, embedWeights, vocab, dim, @"embed");
        MPSGraphTensor *logits = graphGEMV(graph, finalNorm, embedW, @"logits");

        // Compile the graph
        MPSGraphCompilationDescriptor *compDesc = [[MPSGraphCompilationDescriptor alloc] init];

        // Build feeds dictionary: map placeholder tensor to its shape+type.
        MPSGraphShapedType *xType = [[MPSGraphShapedType alloc] initWithShape:@[@1, @(dim)] dataType:MPSDataTypeFloat32];
        NSDictionary<MPSGraphTensor*, MPSGraphShapedType*> *feeds = @{x: xType};

        NSArray<MPSGraphTensor*> *targets = @[logits];

        MPSGraphExecutable *executable = [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                                           feeds:feeds
                                                    targetTensors:targets
                                                 targetOperations:nil
                                            compilationDescriptor:compDesc];

        if (executable == nil) {
            NSLog(@"MPSGraph compilation failed");
            return NULL;
        }

        MPSGraphTransformer *t = (MPSGraphTransformer *)calloc(1, sizeof(MPSGraphTransformer));
        t->graph = (__bridge_retained void *)graph;
        t->executable = (__bridge_retained void *)executable;
        t->device = (__bridge_retained void *)device;
        t->cmdQueue = (__bridge_retained void *)queue;
        t->nLayers = nLayers;
        t->dim = dim;
        t->qDim = qDim;
        t->kvDim = kvDim;
        t->hidden = hidden;
        t->heads = heads;
        t->kvHeads = kvHeads;
        t->headDim = headDim;
        t->vocab = vocab;
        return t;
    }
}

// Execute the compiled graph: input x[dim] -> output logits[vocab]
int mpsGraphTransformerExec(MPSGraphTransformer *t, float *logits, const float *x) {
    if (t == NULL || t->executable == NULL) return -1;
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)t->device;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)t->cmdQueue;
        MPSGraphExecutable *executable = (__bridge MPSGraphExecutable *)t->executable;

        // Create input tensor data
        NSData *xData = [NSData dataWithBytes:x length:(NSUInteger)t->dim * sizeof(float)];
        MPSGraphTensorData *xTD = [[MPSGraphTensorData alloc] initWithMTLBuffer:
            [device newBufferWithBytes:x length:(NSUInteger)t->dim * sizeof(float) options:MTLResourceStorageModeShared]
            shape:@[@1, @(t->dim)] dataType:MPSDataTypeFloat32];

        // Execute
        NSArray<MPSGraphTensorData *> *results = [executable runWithMTLCommandQueue:queue
                                                                         inputsArray:@[xTD]
                                                                        resultsArray:nil
                                                               executionDescriptor:nil];

        if (results.count == 0) return -1;

        // Copy output
        MPSGraphTensorData *outTD = results[0];
        MPSNDArray *outArr = outTD.mpsndarray;
        [outArr readBytes:logits strideBytes:nil];

        return 0;
    }
}

void mpsGraphTransformerDestroy(MPSGraphTransformer *t) {
    if (t != NULL) free(t);
}
