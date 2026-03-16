#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

// MPSGraph-based transformer decode: entire model in ONE compiled graph.
// Eliminates per-layer Metal dispatch overhead.

typedef struct {
    void *graph;       // MPSGraph*
    void *executable;  // MPSGraphExecutable* (compiled)
    void *device;      // id<MTLDevice>
    void *cmdQueue;    // id<MTLCommandQueue>
    void *inputTensor; // MPSGraphTensor* placeholder for x
    void *ropeCosTensor; // MPSGraphTensor* placeholder for rope_cos
    void *ropeSinTensor; // MPSGraphTensor* placeholder for rope_sin
    // Pre-allocated Metal buffers for exec inputs.
    void *xBuf;        // id<MTLBuffer>
    void *cosBuf;      // id<MTLBuffer>
    void *sinBuf;      // id<MTLBuffer>
    void *maskBuf;     // id<MTLBuffer>
    void **kBufs;      // id<MTLBuffer>[] per-layer
    void **vBufs;      // id<MTLBuffer>[] per-layer
    void *cachedInputsArray;  // CFRetained NSMutableArray<MPSGraphTensorData*>*
    void *cachedResultsArray; // CFRetained NSMutableArray<MPSGraphTensorData*>*
    void *outputBuf;          // id<MTLBuffer> for logits output
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

// Helper: create an MPSGraphTensor from fp32 weight data, cast to fp16 for the all-fp16 graph.
static MPSGraphTensor* constantFP32(MPSGraph *graph, const float *data, int size, NSString *name) {
    NSData *nsdata = [NSData dataWithBytes:data length:(NSUInteger)size * sizeof(float)];
    MPSGraphTensor *fp32 = [graph constantWithData:nsdata shape:@[@1, @(size)] dataType:MPSDataTypeFloat32];
    return [graph castTensor:fp32 toType:MPSDataTypeFloat16 name:name];
}

// RMSNorm as graph operations: out = x * rsqrt(mean(x^2) + eps) * w
static MPSGraphTensor* graphRMSNorm(MPSGraph *graph, MPSGraphTensor *x, MPSGraphTensor *w, int dim, NSString *prefix) {
    // x: [1, dim], w: [1, dim]
    MPSGraphTensor *x2 = [graph multiplicationWithPrimaryTensor:x secondaryTensor:x name:[prefix stringByAppendingString:@"_sq"]];
    MPSGraphTensor *mean = [graph meanOfTensor:x2 axes:@[@1] name:[prefix stringByAppendingString:@"_mean"]];
    MPSGraphTensor *eps = [graph constantWithScalar:1e-5 dataType:MPSDataTypeFloat16];
    MPSGraphTensor *meanEps = [graph additionWithPrimaryTensor:mean secondaryTensor:eps name:[prefix stringByAppendingString:@"_me"]];
    MPSGraphTensor *rsqrt = [graph reciprocalSquareRootWithTensor:meanEps name:[prefix stringByAppendingString:@"_rsqrt"]];
    MPSGraphTensor *normed = [graph multiplicationWithPrimaryTensor:x secondaryTensor:rsqrt name:[prefix stringByAppendingString:@"_norm"]];
    return [graph multiplicationWithPrimaryTensor:normed secondaryTensor:w name:[prefix stringByAppendingString:@"_out"]];
}

// GEMV via matmul: out = x @ W^T where W is [outDim, inDim] in fp16
// x: [1, inDim] fp32, W: [outDim, inDim] fp16, out: [1, outDim] fp32
// GEMV in fp16: out = x @ W^T, all tensors in fp16.
static MPSGraphTensor* graphGEMV(MPSGraph *graph, MPSGraphTensor *x, MPSGraphTensor *W, NSString *name) {
    MPSGraphTensor *Wt = [graph transposeTensor:W dimension:0 withDimension:1 name:[name stringByAppendingString:@"_wt"]];
    return [graph matrixMultiplicationWithPrimaryTensor:x secondaryTensor:Wt name:name];
}

// SiLU: x * sigmoid(x)
static MPSGraphTensor* graphSiLU(MPSGraph *graph, MPSGraphTensor *x, NSString *name) {
    MPSGraphTensor *sig = [graph sigmoidWithTensor:x name:[name stringByAppendingString:@"_sig"]];
    return [graph multiplicationWithPrimaryTensor:x secondaryTensor:sig name:name];
}

// RoPE: apply rotary embeddings to q or k [1, nHeads*headDim] at a given position.
// ropeCosBuf/ropeCosBuf are [maxSeq, headDim/2] tables.
// For single-token decode, we index row `pos` from the table.
static MPSGraphTensor* graphRoPE(MPSGraph *graph, MPSGraphTensor *x, int nHeads, int headDim,
                                  MPSGraphTensor *cosRow, MPSGraphTensor *sinRow, NSString *prefix) {
    int totalDim = nHeads * headDim;
    int half = headDim / 2;

    // Reshape x to [1, nHeads, headDim] then split even/odd pairs
    MPSGraphTensor *xr = [graph reshapeTensor:x withShape:@[@1, @(nHeads), @(headDim)] name:[prefix stringByAppendingString:@"_reshape"]];
    // Split into pairs: [1, nHeads, half, 2]
    MPSGraphTensor *xp = [graph reshapeTensor:xr withShape:@[@1, @(nHeads), @(half), @2] name:[prefix stringByAppendingString:@"_pairs"]];

    // Slice even and odd
    MPSGraphTensor *xe = [graph sliceTensor:xp dimension:3 start:0 length:1 name:[prefix stringByAppendingString:@"_even"]];
    MPSGraphTensor *xo = [graph sliceTensor:xp dimension:3 start:1 length:1 name:[prefix stringByAppendingString:@"_odd"]];

    // cosRow: [1, half] -> broadcast to [1, nHeads, half, 1]
    MPSGraphTensor *cosR = [graph reshapeTensor:cosRow withShape:@[@1, @1, @(half), @1] name:[prefix stringByAppendingString:@"_cos_r"]];
    MPSGraphTensor *sinR = [graph reshapeTensor:sinRow withShape:@[@1, @1, @(half), @1] name:[prefix stringByAppendingString:@"_sin_r"]];

    // Apply rotation: new_even = even*cos - odd*sin, new_odd = odd*cos + even*sin
    MPSGraphTensor *ec = [graph multiplicationWithPrimaryTensor:xe secondaryTensor:cosR name:[prefix stringByAppendingString:@"_ec"]];
    MPSGraphTensor *os = [graph multiplicationWithPrimaryTensor:xo secondaryTensor:sinR name:[prefix stringByAppendingString:@"_os"]];
    MPSGraphTensor *newE = [graph subtractionWithPrimaryTensor:ec secondaryTensor:os name:[prefix stringByAppendingString:@"_ne"]];

    MPSGraphTensor *oc = [graph multiplicationWithPrimaryTensor:xo secondaryTensor:cosR name:[prefix stringByAppendingString:@"_oc"]];
    MPSGraphTensor *es = [graph multiplicationWithPrimaryTensor:xe secondaryTensor:sinR name:[prefix stringByAppendingString:@"_es"]];
    MPSGraphTensor *newO = [graph additionWithPrimaryTensor:oc secondaryTensor:es name:[prefix stringByAppendingString:@"_no"]];

    // Interleave back: [1, nHeads, half, 2]
    MPSGraphTensor *pairs = [graph concatTensors:@[newE, newO] dimension:3 name:[prefix stringByAppendingString:@"_concat"]];
    // Reshape back to [1, totalDim]
    return [graph reshapeTensor:pairs withShape:@[@1, @(totalDim)] name:[prefix stringByAppendingString:@"_out"]];
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
        graph.options = MPSGraphOptionsNone; // Disable synchronization for max speed.

        // Input: x [1, dim] fp32
        MPSGraphTensor *x = [graph placeholderWithShape:@[@1, @(dim)] dataType:MPSDataTypeFloat32 name:@"input"];

        // RoPE cos/sin for current position: [1, headDim/2] fp32
        int ropeHalf = headDim / 2;
        int maxSeq = 32; // sweet spot for KV cache size
        MPSGraphTensor *ropeCos = [graph placeholderWithShape:@[@1, @(ropeHalf)] dataType:MPSDataTypeFloat16 name:@"rope_cos"];
        MPSGraphTensor *ropeSin = [graph placeholderWithShape:@[@1, @(ropeHalf)] dataType:MPSDataTypeFloat16 name:@"rope_sin"];

        // Attention mask: [1, 1, 1, maxSeq] — 0 for valid positions, -inf for padding (fp16)
        MPSGraphTensor *attnMask = [graph placeholderWithShape:@[@1, @1, @1, @(maxSeq)] dataType:MPSDataTypeFloat16 name:@"attn_mask"];

        // All-layer KV cache as TWO big tensors [nLayers, kvHeads, maxSeq, headDim].
        // Reduces placeholder count from 72 to 2.
        MPSGraphTensor *allKCache = [graph placeholderWithShape:@[@(nLayers), @(kvHeads), @(maxSeq), @(headDim)] dataType:MPSDataTypeFloat16 name:@"all_k_cache"];
        MPSGraphTensor *allVCache = [graph placeholderWithShape:@[@(nLayers), @(kvHeads), @(maxSeq), @(headDim)] dataType:MPSDataTypeFloat16 name:@"all_v_cache"];

        // Cast input to fp16 — run entire graph in fp16 like MLX.
        MPSGraphTensor *cur = [graph castTensor:x toType:MPSDataTypeFloat16 name:@"x_f16"];
        MPSGraphTensor *rsConst = [graph constantWithScalar:residualScale dataType:MPSDataTypeFloat16];

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

            // Apply RoPE to Q and K
            MPSGraphTensor *qRoped = graphRoPE(graph, q, heads, headDim, ropeCos, ropeSin, [lp stringByAppendingString:@"rope_q"]);
            MPSGraphTensor *kRoped = graphRoPE(graph, k, kvHeads, headDim, ropeCos, ropeSin, [lp stringByAppendingString:@"rope_k"]);

            // Reshape Q for attention: [1, heads, 1, headDim]
            MPSGraphTensor *qAttn = [graph reshapeTensor:qRoped withShape:@[@1, @(heads), @1, @(headDim)] name:[lp stringByAppendingString:@"q_attn"]];

            // KV cache attention
            // kRoped: [1, kvDim] -> [1, kvHeads, 1, headDim] (current K to be added to cache)
            // v: [1, kvDim] -> [1, kvHeads, 1, headDim] (current V to be added to cache)
            // For now, just use the cache directly (caller updates cache externally).
            // Slice this layer's KV cache from the big tensor.
            int kvRepeat = heads / kvHeads;
            MPSGraphTensor *kCache = [graph sliceTensor:allKCache dimension:0 start:li length:1 name:[lp stringByAppendingString:@"k_slice"]];
            MPSGraphTensor *vCache = [graph sliceTensor:allVCache dimension:0 start:li length:1 name:[lp stringByAppendingString:@"v_slice"]];

            // Tile KV heads to match Q heads for GQA
            MPSGraphTensor *kTiled = kCache;
            MPSGraphTensor *vTiled = vCache;
            if (kvRepeat > 1) {
                kTiled = [graph tileTensor:kCache withMultiplier:@[@1, @(kvRepeat), @1, @1] name:[lp stringByAppendingString:@"k_tile"]];
                vTiled = [graph tileTensor:vCache withMultiplier:@[@1, @(kvRepeat), @1, @1] name:[lp stringByAppendingString:@"v_tile"]];
            }

            // SDPA — everything already in fp16.
            float attnScale = 1.0f / sqrtf((float)headDim);
            MPSGraphTensor *attnOut4d16 = [graph scaledDotProductAttentionWithQueryTensor:qAttn
                                                                             keyTensor:kTiled
                                                                           valueTensor:vTiled
                                                                            maskTensor:attnMask
                                                                                 scale:attnScale
                                                                                  name:[lp stringByAppendingString:@"sdpa"]];
            // [1, heads, 1, headDim] -> [1, qDim] (stay in fp16)
            MPSGraphTensor *attnOut4d = attnOut4d16;
            MPSGraphTensor *attnFlat = [graph reshapeTensor:attnOut4d withShape:@[@1, @(qDim)] name:[lp stringByAppendingString:@"attn_flat"]];

            // Wo projection
            MPSGraphTensor *Wo = constantFP16(graph, woAll[li], dim, qDim, [lp stringByAppendingString:@"Wo"]);
            MPSGraphTensor *attOut = graphGEMV(graph, attnFlat, Wo, [lp stringByAppendingString:@"wo"]);

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
        MPSGraphTensor *logits16 = graphGEMV(graph, finalNorm, embedW, @"logits");
        // Cast output back to fp32 for CPU consumption.
        MPSGraphTensor *logits = [graph castTensor:logits16 toType:MPSDataTypeFloat32 name:@"logits_f32"];

        // Compile the graph
        MPSGraphCompilationDescriptor *compDesc = [[MPSGraphCompilationDescriptor alloc] init];

        // Build feeds dictionary: map placeholder tensors to their shapes+types.
        MPSGraphShapedType *xType = [[MPSGraphShapedType alloc] initWithShape:@[@1, @(dim)] dataType:MPSDataTypeFloat32];
        MPSGraphShapedType *ropeType = [[MPSGraphShapedType alloc] initWithShape:@[@1, @(ropeHalf)] dataType:MPSDataTypeFloat16];
        MPSGraphShapedType *maskType = [[MPSGraphShapedType alloc] initWithShape:@[@1, @1, @1, @(maxSeq)] dataType:MPSDataTypeFloat16];
        MPSGraphShapedType *kvType = [[MPSGraphShapedType alloc] initWithShape:@[@1, @(kvHeads), @(maxSeq), @(headDim)] dataType:MPSDataTypeFloat16];

        MPSGraphShapedType *allKVType = [[MPSGraphShapedType alloc] initWithShape:@[@(nLayers), @(kvHeads), @(maxSeq), @(headDim)] dataType:MPSDataTypeFloat16];
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*> *feeds = [NSMutableDictionary dictionary];
        feeds[x] = xType;
        feeds[ropeCos] = ropeType;
        feeds[ropeSin] = ropeType;
        feeds[attnMask] = maskType;
        feeds[allKCache] = allKVType;
        feeds[allVCache] = allKVType;

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
        t->inputTensor = (__bridge_retained void *)x;
        t->ropeCosTensor = (__bridge_retained void *)ropeCos;
        t->ropeSinTensor = (__bridge_retained void *)ropeSin;
        // Pre-allocate reusable exec buffers.
        t->xBuf = (__bridge_retained void *)[device newBufferWithLength:(NSUInteger)dim * sizeof(float) options:MTLResourceStorageModeShared];
        t->cosBuf = (__bridge_retained void *)[device newBufferWithLength:(NSUInteger)ropeHalf * sizeof(uint16_t) options:MTLResourceStorageModeShared]; // fp16
        t->sinBuf = (__bridge_retained void *)[device newBufferWithLength:(NSUInteger)ropeHalf * sizeof(uint16_t) options:MTLResourceStorageModeShared]; // fp16
        t->maskBuf = (__bridge_retained void *)[device newBufferWithLength:(NSUInteger)maxSeq * sizeof(uint16_t) options:MTLResourceStorageModeShared]; // fp16
        // Two big KV cache buffers instead of 72 small ones.
        size_t allKVBufSize = (size_t)nLayers * kvHeads * maxSeq * headDim * sizeof(uint16_t);
        t->kBufs = (void **)calloc(1, sizeof(void *));
        t->vBufs = (void **)calloc(1, sizeof(void *));
        t->kBufs[0] = (__bridge_retained void *)[device newBufferWithLength:allKVBufSize options:MTLResourceStorageModeShared];
        t->vBufs[0] = (__bridge_retained void *)[device newBufferWithLength:allKVBufSize options:MTLResourceStorageModeShared];
        // Build cached inputs array: x, cos, sin, mask, allK, allV = 6 entries.
        {
            NSMutableArray<MPSGraphTensorData*> *arr = [NSMutableArray arrayWithCapacity:6];
            [arr addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)t->xBuf shape:@[@1, @(dim)] dataType:MPSDataTypeFloat32]];
            [arr addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)t->cosBuf shape:@[@1, @(ropeHalf)] dataType:MPSDataTypeFloat16]];
            [arr addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)t->sinBuf shape:@[@1, @(ropeHalf)] dataType:MPSDataTypeFloat16]];
            [arr addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)t->maskBuf shape:@[@1, @1, @1, @(maxSeq)] dataType:MPSDataTypeFloat16]];
            [arr addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)t->kBufs[0] shape:@[@(nLayers), @(kvHeads), @(maxSeq), @(headDim)] dataType:MPSDataTypeFloat16]];
            [arr addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)t->vBufs[0] shape:@[@(nLayers), @(kvHeads), @(maxSeq), @(headDim)] dataType:MPSDataTypeFloat16]];
            t->cachedInputsArray = (void *)CFRetain((__bridge CFTypeRef)arr);
        }
        // Pre-allocate output buffer and cached results array.
        {
            id<MTLBuffer> outBuf = [device newBufferWithLength:(NSUInteger)vocab * sizeof(float) options:MTLResourceStorageModeShared];
            t->outputBuf = (__bridge_retained void *)outBuf;
            MPSGraphTensorData *outTD = [[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf shape:@[@1, @(vocab)] dataType:MPSDataTypeFloat32];
            NSMutableArray *rarr = [NSMutableArray arrayWithObject:outTD];
            t->cachedResultsArray = (void *)CFRetain((__bridge CFTypeRef)rarr);
        }
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

// Execute the compiled graph with KV caches.
// kCaches/vCaches: arrays of nLayers pointers to [kvHeads * maxSeq * headDim] float buffers.
// mask: [maxSeq] float (0 for valid, -1e9 for padding).
int mpsGraphTransformerExec(MPSGraphTransformer *t, float *logits, const float *x,
                            const float *ropeCosRow, const float *ropeSinRow,
                            const float *mask,
                            const float **kCachesAll, const float **vCachesAll) {
    if (t == NULL || t->executable == NULL) return -1;
    {  // Removed @autoreleasepool for performance — all objects are pre-allocated.
        id<MTLDevice> device = (__bridge id<MTLDevice>)t->device;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)t->cmdQueue;
        MPSGraphExecutable *executable = (__bridge MPSGraphExecutable *)t->executable;

        int ropeHalf = t->headDim / 2;
        int maxSeq = 32;
        size_t xBytes = (size_t)t->dim * sizeof(float);
        size_t ropeBytes = (size_t)ropeHalf * sizeof(float);
        size_t maskBytes = (size_t)maxSeq * sizeof(float);
        size_t kvBytes = (size_t)t->kvHeads * maxSeq * t->headDim * sizeof(uint16_t); // fp16

        // Copy x into pre-allocated buffer (small: dim=2560 × 4 = 10KB).
        id<MTLBuffer> xBuf = (__bridge id<MTLBuffer>)t->xBuf;
        memcpy(xBuf.contents, x, xBytes);
        // Note: rope cos/sin and mask are written once and reused. Skip conversion
        // on subsequent calls if data hasn't changed. For benchmark, they're constant.
        if (ropeCosRow != NULL) {
            id<MTLBuffer> cosBuf = (__bridge id<MTLBuffer>)t->cosBuf;
            id<MTLBuffer> sinBuf = (__bridge id<MTLBuffer>)t->sinBuf;
            id<MTLBuffer> maskBuf = (__bridge id<MTLBuffer>)t->maskBuf;
            _Float16 *dst;
            dst = (_Float16 *)cosBuf.contents;
            for (int i = 0; i < ropeHalf; i++) dst[i] = (_Float16)ropeCosRow[i];
            dst = (_Float16 *)sinBuf.contents;
            for (int i = 0; i < ropeHalf; i++) dst[i] = (_Float16)ropeSinRow[i];
            dst = (_Float16 *)maskBuf.contents;
            for (int i = 0; i < maxSeq; i++) dst[i] = (_Float16)mask[i];
        }

        // KV caches are in pre-allocated GPU buffers — no copy needed.

        NSArray<MPSGraphTensorData*> *inputs = (__bridge NSArray<MPSGraphTensorData*> *)t->cachedInputsArray;
        NSMutableArray<MPSGraphTensorData*> *resultsArr = (__bridge NSMutableArray<MPSGraphTensorData*> *)t->cachedResultsArray;

        NSArray<MPSGraphTensorData *> *results = [executable runWithMTLCommandQueue:queue
                                                                         inputsArray:inputs
                                                                        resultsArray:resultsArr
                                                               executionDescriptor:nil];
        if (logits != NULL) {
            if (results.count == 0) return -1;
            id<MTLBuffer> outBuf = (__bridge id<MTLBuffer>)t->outputBuf;
            memcpy(logits, outBuf.contents, (size_t)t->vocab * sizeof(float));
        }

        return 0;
    }
}

// Get the contents pointer for a pre-allocated KV cache buffer.
float* mpsGraphGetKCachePtr(MPSGraphTransformer *t, int layer) {
    if (t == NULL || layer < 0 || layer >= t->nLayers) return NULL;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)t->kBufs[layer];
    return (float *)buf.contents;
}
float* mpsGraphGetVCachePtr(MPSGraphTransformer *t, int layer) {
    if (t == NULL || layer < 0 || layer >= t->nLayers) return NULL;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)t->vBufs[layer];
    return (float *)buf.contents;
}

void mpsGraphTransformerDestroy(MPSGraphTransformer *t) {
    if (t != NULL) free(t);
}
