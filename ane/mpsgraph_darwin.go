//go:build darwin && cgo

package ane

/*
#cgo darwin LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation

#include <stdint.h>

typedef struct MPSGraphTransformer MPSGraphTransformer;

extern MPSGraphTransformer* mpsGraphTransformerCreate(
    int nLayers, int dim, int qDim, int kvDim, int hidden,
    int heads, int kvHeads, int headDim, int vocab,
    const float *rmsFinalWeights,
    const uint16_t **wqAll, const uint16_t **wkAll, const uint16_t **wvAll, const uint16_t **woAll,
    const uint16_t **w1All, const uint16_t **w3All, const uint16_t **w2All,
    const float **rmsAttAll, const float **rmsFFNAll,
    const uint16_t *embedWeights,
    float residualScale
);
extern int mpsGraphTransformerExec(MPSGraphTransformer *t, float *logits, const float *x,
                                   const float *ropeCosRow, const float *ropeSinRow,
                                   const float *mask,
                                   const float **kCachesAll, const float **vCachesAll);
extern float* mpsGraphGetKCachePtr(MPSGraphTransformer *t, int layer);
extern float* mpsGraphGetVCachePtr(MPSGraphTransformer *t, int layer);
extern void mpsGraphTransformerDestroy(MPSGraphTransformer *t);
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

// MPSGraphDecoder is a compiled GPU compute graph for single-token decode.
// The entire transformer (36 layers of matmuls + activations) runs as ONE
// optimized GPU dispatch — no per-layer Metal command buffer overhead.
type MPSGraphDecoder struct {
	handle *C.MPSGraphTransformer
	// fp16 weight arrays (kept alive for GPU references)
	wq, wk, wv, wo, w1, w3, w2 [][]uint16
	embedFP16                   []uint16
}

// NewMPSGraphDecoder compiles the full transformer as an MPSGraph.
// This is expensive (minutes) but only done once. After compilation,
// each Exec() call runs the entire model in ~15ms on GPU.
func NewMPSGraphDecoder(cfg stories.ModelConfig, mw *stories.ModelWeights) (*MPSGraphDecoder, error) {
	nLayers := cfg.NLayers
	dim := cfg.Dim
	qDim := cfg.QDim()
	kvDim := cfg.KVDim()
	hidden := cfg.Hidden
	heads := cfg.Heads
	kvHeads := cfg.EffectiveKVHeads()
	headDim := cfg.HeadDim()
	vocab := cfg.Vocab
	residualScale := float32(cfg.ResidualScale())

	d := &MPSGraphDecoder{
		wq: make([][]uint16, nLayers),
		wk: make([][]uint16, nLayers),
		wv: make([][]uint16, nLayers),
		wo: make([][]uint16, nLayers),
		w1: make([][]uint16, nLayers),
		w3: make([][]uint16, nLayers),
		w2: make([][]uint16, nLayers),
	}

	// Convert all weights to fp16
	wqPtrs := make([]*C.uint16_t, nLayers)
	wkPtrs := make([]*C.uint16_t, nLayers)
	wvPtrs := make([]*C.uint16_t, nLayers)
	woPtrs := make([]*C.uint16_t, nLayers)
	w1Ptrs := make([]*C.uint16_t, nLayers)
	w3Ptrs := make([]*C.uint16_t, nLayers)
	w2Ptrs := make([]*C.uint16_t, nLayers)
	rmsAttPtrs := make([]*C.float, nLayers)
	rmsFFNPtrs := make([]*C.float, nLayers)

	for i := 0; i < nLayers; i++ {
		layer := mw.Layers[i]
		d.wq[i] = make([]uint16, len(layer.Wq))
		convertF32ToFP16(d.wq[i], layer.Wq)
		wqPtrs[i] = (*C.uint16_t)(unsafe.Pointer(&d.wq[i][0]))

		d.wk[i] = make([]uint16, len(layer.Wk))
		convertF32ToFP16(d.wk[i], layer.Wk)
		wkPtrs[i] = (*C.uint16_t)(unsafe.Pointer(&d.wk[i][0]))

		d.wv[i] = make([]uint16, len(layer.Wv))
		convertF32ToFP16(d.wv[i], layer.Wv)
		wvPtrs[i] = (*C.uint16_t)(unsafe.Pointer(&d.wv[i][0]))

		d.wo[i] = make([]uint16, len(layer.Wo))
		convertF32ToFP16(d.wo[i], layer.Wo)
		woPtrs[i] = (*C.uint16_t)(unsafe.Pointer(&d.wo[i][0]))

		d.w1[i] = make([]uint16, len(layer.W1))
		convertF32ToFP16(d.w1[i], layer.W1)
		w1Ptrs[i] = (*C.uint16_t)(unsafe.Pointer(&d.w1[i][0]))

		d.w3[i] = make([]uint16, len(layer.W3))
		convertF32ToFP16(d.w3[i], layer.W3)
		w3Ptrs[i] = (*C.uint16_t)(unsafe.Pointer(&d.w3[i][0]))

		d.w2[i] = make([]uint16, len(layer.W2))
		convertF32ToFP16(d.w2[i], layer.W2)
		w2Ptrs[i] = (*C.uint16_t)(unsafe.Pointer(&d.w2[i][0]))

		rmsAttPtrs[i] = (*C.float)(unsafe.Pointer(&layer.RMSAtt[0]))
		rmsFFNPtrs[i] = (*C.float)(unsafe.Pointer(&layer.RMSFFN[0]))
	}

	d.embedFP16 = make([]uint16, len(mw.Embed))
	convertF32ToFP16(d.embedFP16, mw.Embed)

	// Pin all Go pointers for CGo.
	var pinner runtime.Pinner
	defer pinner.Unpin()
	for i := 0; i < nLayers; i++ {
		pinner.Pin(&d.wq[i][0])
		pinner.Pin(&d.wk[i][0])
		pinner.Pin(&d.wv[i][0])
		pinner.Pin(&d.wo[i][0])
		pinner.Pin(&d.w1[i][0])
		pinner.Pin(&d.w3[i][0])
		pinner.Pin(&d.w2[i][0])
		pinner.Pin(&mw.Layers[i].RMSAtt[0])
		pinner.Pin(&mw.Layers[i].RMSFFN[0])
	}
	pinner.Pin(&wqPtrs[0])
	pinner.Pin(&wkPtrs[0])
	pinner.Pin(&wvPtrs[0])
	pinner.Pin(&woPtrs[0])
	pinner.Pin(&w1Ptrs[0])
	pinner.Pin(&w3Ptrs[0])
	pinner.Pin(&w2Ptrs[0])
	pinner.Pin(&rmsAttPtrs[0])
	pinner.Pin(&rmsFFNPtrs[0])
	pinner.Pin(&d.embedFP16[0])
	pinner.Pin(&mw.RMSFinal[0])

	h := C.mpsGraphTransformerCreate(
		C.int(nLayers), C.int(dim), C.int(qDim), C.int(kvDim), C.int(hidden),
		C.int(heads), C.int(kvHeads), C.int(headDim), C.int(vocab),
		(*C.float)(unsafe.Pointer(&mw.RMSFinal[0])),
		(**C.uint16_t)(unsafe.Pointer(&wqPtrs[0])),
		(**C.uint16_t)(unsafe.Pointer(&wkPtrs[0])),
		(**C.uint16_t)(unsafe.Pointer(&wvPtrs[0])),
		(**C.uint16_t)(unsafe.Pointer(&woPtrs[0])),
		(**C.uint16_t)(unsafe.Pointer(&w1Ptrs[0])),
		(**C.uint16_t)(unsafe.Pointer(&w3Ptrs[0])),
		(**C.uint16_t)(unsafe.Pointer(&w2Ptrs[0])),
		(**C.float)(unsafe.Pointer(&rmsAttPtrs[0])),
		(**C.float)(unsafe.Pointer(&rmsFFNPtrs[0])),
		(*C.uint16_t)(unsafe.Pointer(&d.embedFP16[0])),
		C.float(residualScale),
	)
	if h == nil {
		return nil, fmt.Errorf("MPSGraph transformer compilation failed")
	}
	d.handle = h
	return d, nil
}

// Exec runs the compiled graph with KV caches.
// kCaches/vCaches: per-layer KV cache buffers [kvHeads * maxSeq * headDim].
// mask: [maxSeq] attention mask (0 valid, -1e9 padding).
func (d *MPSGraphDecoder) Exec(logits, x, ropeCosRow, ropeSinRow, mask []float32,
	kCaches, vCaches [][]float32) error {
	if d == nil || d.handle == nil {
		return fmt.Errorf("MPSGraph decoder not initialized")
	}
	nLayers := len(kCaches)
	kPtrs := make([]*C.float, nLayers)
	vPtrs := make([]*C.float, nLayers)
	var pinner runtime.Pinner
	defer pinner.Unpin()
	for i := 0; i < nLayers; i++ {
		pinner.Pin(&kCaches[i][0])
		pinner.Pin(&vCaches[i][0])
		kPtrs[i] = (*C.float)(unsafe.Pointer(&kCaches[i][0]))
		vPtrs[i] = (*C.float)(unsafe.Pointer(&vCaches[i][0]))
	}
	pinner.Pin(&kPtrs[0])
	pinner.Pin(&vPtrs[0])
	pinner.Pin(&mask[0])

	if C.mpsGraphTransformerExec(d.handle,
		(*C.float)(unsafe.Pointer(&logits[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&ropeCosRow[0])),
		(*C.float)(unsafe.Pointer(&ropeSinRow[0])),
		(*C.float)(unsafe.Pointer(&mask[0])),
		(**C.float)(unsafe.Pointer(&kPtrs[0])),
		(**C.float)(unsafe.Pointer(&vPtrs[0])),
	) != 0 {
		return fmt.Errorf("MPSGraph execution failed")
	}
	return nil
}

// KVCacheSlice returns a Go slice backed by the GPU-resident KV cache buffer for a layer.
// Writes to this slice go directly to GPU memory (unified memory).
func (d *MPSGraphDecoder) KCacheSlice(layer, size int) []float32 {
	ptr := C.mpsGraphGetKCachePtr(d.handle, C.int(layer))
	if ptr == nil {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), size)
}

func (d *MPSGraphDecoder) VCacheSlice(layer, size int) []float32 {
	ptr := C.mpsGraphGetVCachePtr(d.handle, C.int(layer))
	if ptr == nil {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), size)
}

// ExecNoCopy runs without copying logits output (for benchmarking pure GPU throughput).
func (d *MPSGraphDecoder) ExecNoCopy(x, ropeCosRow, ropeSinRow, mask []float32) error {
	if d == nil || d.handle == nil {
		return fmt.Errorf("MPSGraph decoder not initialized")
	}
	// Copy only the small inputs (x, cos, sin, mask).
	id := d.handle
	xBuf := (*C.float)(unsafe.Pointer(&x[0]))
	cosBuf := (*C.float)(unsafe.Pointer(&ropeCosRow[0]))
	sinBuf := (*C.float)(unsafe.Pointer(&ropeSinRow[0]))
	maskBuf := (*C.float)(unsafe.Pointer(&mask[0]))
	if C.mpsGraphTransformerExec(id, nil, xBuf, cosBuf, sinBuf, maskBuf, nil, nil) != 0 {
		return fmt.Errorf("MPSGraph execution failed")
	}
	return nil
}

// ExecZeroCopy runs without copying KV caches (they're already in GPU buffers).
func (d *MPSGraphDecoder) ExecZeroCopy(logits, x, ropeCosRow, ropeSinRow, mask []float32) error {
	if d == nil || d.handle == nil {
		return fmt.Errorf("MPSGraph decoder not initialized")
	}
	pinner := runtime.Pinner{}
	defer pinner.Unpin()
	pinner.Pin(&mask[0])

	if C.mpsGraphTransformerExec(d.handle,
		(*C.float)(unsafe.Pointer(&logits[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&ropeCosRow[0])),
		(*C.float)(unsafe.Pointer(&ropeSinRow[0])),
		(*C.float)(unsafe.Pointer(&mask[0])),
		nil, nil, // NULL = skip KV cache copy, use pre-allocated buffers
	) != 0 {
		return fmt.Errorf("MPSGraph execution failed")
	}
	return nil
}

// Close releases GPU resources.
func (d *MPSGraphDecoder) Close() {
	if d != nil && d.handle != nil {
		C.mpsGraphTransformerDestroy(d.handle)
		d.handle = nil
	}
}
