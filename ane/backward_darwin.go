//go:build darwin

package ane

import (
	"fmt"

	"github.com/tmc/apple/x/ane/mil"
	"github.com/tmc/apple/x/ane/model"
	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

type layerBackward struct {
	dim     int
	hidden  int
	heads   int
	seq     int
	scoreCh int

	ffnW2 *model.Kernel
	ffn   *model.Kernel
	wot   *model.Kernel
	sdpa1 *model.Kernel
	sdpa2 *model.Kernel
	qkv   *model.Kernel

	metrics  *aneStepMetrics
	dynamic  bool
	ffnIn    []float32
	ffnOut   []float32
	sdpa1In  []float32
	sdpa1Out []float32
	sdpa2In  []float32
	sdpa2Out []float32
	qkvIn    []float32
}

var compileStoriesLayerBackwardFunc = compileStoriesLayerBackward

func compileStoriesLayerBackward(layer stories.LayerWeights, seq int) (*layerBackward, error) {
	if lb, err := compileStoriesLayerBackwardDynamic(layer, seq); err == nil {
		return lb, nil
	}
	const (
		dim    = stories.Dim
		hidden = stories.Hidden
		heads  = stories.Heads
	)
	if dim <= 0 || hidden <= 0 || heads <= 0 || seq <= 0 {
		return nil, fmt.Errorf("compile layer backward: invalid shape dim=%d hidden=%d heads=%d seq=%d", dim, hidden, heads, seq)
	}
	if err := validateLayerWeights(dim, hidden, layerForwardWeights{
		Wq: layer.Wq,
		Wk: layer.Wk,
		Wv: layer.Wv,
		Wo: layer.Wo,
		W1: layer.W1,
		W2: layer.W2,
	}); err != nil {
		return nil, err
	}

	w2tBlob, err := mil.BuildTransposedWeightBlob(layer.W2, dim, hidden)
	if err != nil {
		return nil, fmt.Errorf("compile layer backward: w2t blob: %w", err)
	}
	w1tBlob, err := mil.BuildTransposedWeightBlob(layer.W1, hidden, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer backward: w1t blob: %w", err)
	}
	wotBlob, err := mil.BuildTransposedWeightBlob(layer.Wo, dim, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer backward: wot blob: %w", err)
	}
	maskBlob, err := mil.BuildCausalMaskBlob(seq)
	if err != nil {
		return nil, fmt.Errorf("compile layer backward: mask blob: %w", err)
	}
	wqtBlob, err := mil.BuildTransposedWeightBlob(layer.Wq, dim, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer backward: wqt blob: %w", err)
	}
	wktBlob, err := mil.BuildTransposedWeightBlob(layer.Wk, dim, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer backward: wkt blob: %w", err)
	}
	wvtBlob, err := mil.BuildTransposedWeightBlob(layer.Wv, dim, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer backward: wvt blob: %w", err)
	}

	ffn, err := compileMultiFP16Kernel(
		mil.GenFFNBackward(dim, hidden, seq),
		[]model.WeightFile{
			{Path: "@model_path/weights/w2t.bin", Blob: w2tBlob},
			{Path: "@model_path/weights/w1t.bin", Blob: w1tBlob},
		},
		dim+hidden,
		dim+hidden,
		seq,
	)
	if err != nil {
		return nil, fmt.Errorf("compile layer backward: ffn: %w", err)
	}
	defer func() {
		if err != nil {
			closeKernel(ffn)
		}
	}()

	scoreCh := heads * seq
	sdpa1, err := compileMultiFP16Kernel(
		mil.GenSDPABackward1(dim, heads, seq),
		[]model.WeightFile{
			{Path: "@model_path/weights/wot.bin", Blob: wotBlob},
			{Path: "@model_path/weights/mask.bin", Blob: maskBlob},
		},
		4*dim,
		dim+2*scoreCh,
		seq,
	)
	if err != nil {
		return nil, fmt.Errorf("compile layer backward: sdpa1: %w", err)
	}
	defer func() {
		if err != nil {
			closeKernel(sdpa1)
		}
	}()

	sdpa2, err := compileMultiFP16Kernel(
		mil.GenSDPABackward2(dim, heads, seq),
		nil,
		2*scoreCh+2*dim,
		2*dim,
		seq,
	)
	if err != nil {
		return nil, fmt.Errorf("compile layer backward: sdpa2: %w", err)
	}
	defer func() {
		if err != nil {
			closeKernel(sdpa2)
		}
	}()

	qkv, err := compileMultiFP16Kernel(
		mil.GenQKVBackward(dim, heads, seq),
		[]model.WeightFile{
			{Path: "@model_path/weights/wqt.bin", Blob: wqtBlob},
			{Path: "@model_path/weights/wkt.bin", Blob: wktBlob},
			{Path: "@model_path/weights/wvt.bin", Blob: wvtBlob},
		},
		3*dim,
		dim,
		seq,
	)
	if err != nil {
		return nil, fmt.Errorf("compile layer backward: qkv: %w", err)
	}

	return &layerBackward{
		dim:      dim,
		hidden:   hidden,
		heads:    heads,
		seq:      seq,
		scoreCh:  scoreCh,
		ffn:      ffn,
		sdpa1:    sdpa1,
		sdpa2:    sdpa2,
		qkv:      qkv,
		ffnIn:    make([]float32, (dim+hidden)*seq),
		ffnOut:   make([]float32, (dim+hidden)*seq),
		sdpa1In:  make([]float32, 4*dim*seq),
		sdpa1Out: make([]float32, (dim+2*scoreCh)*seq),
		sdpa2In:  make([]float32, (2*scoreCh+2*dim)*seq),
		sdpa2Out: make([]float32, 2*dim*seq),
		qkvIn:    make([]float32, 3*dim*seq),
	}, nil
}

func (lb *layerBackward) close() {
	if lb == nil {
		return
	}
	closeKernel(lb.ffnW2)
	closeKernel(lb.ffn)
	closeKernel(lb.wot)
	closeKernel(lb.sdpa1)
	closeKernel(lb.sdpa2)
	closeKernel(lb.qkv)
	lb.ffnW2 = nil
	lb.ffn = nil
	lb.wot = nil
	lb.sdpa1 = nil
	lb.sdpa2 = nil
	lb.qkv = nil
	lb.ffnIn = nil
	lb.ffnOut = nil
	lb.sdpa1In = nil
	lb.sdpa1Out = nil
	lb.sdpa2In = nil
	lb.sdpa2Out = nil
	lb.qkvIn = nil
}

func (lb *layerBackward) runFFN(dxNorm, dh1, dFFN, h1 []float32) error {
	if lb != nil && lb.dynamic {
		return lb.runDynamicFFN(dxNorm, dh1, dFFN, h1)
	}
	if lb == nil || lb.ffn == nil {
		return fmt.Errorf("run layer backward ffn: layer is closed")
	}
	dimN := lb.dim * lb.seq
	hiddenN := lb.hidden * lb.seq
	if err := checkLen("run layer backward ffn", "dffn", dFFN, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward ffn", "h1", h1, hiddenN); err != nil {
		return err
	}
	if err := checkLen("run layer backward ffn", "dx", dxNorm, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward ffn", "dh1", dh1, hiddenN); err != nil {
		return err
	}

	concatInto(lb.ffnIn, dFFN, h1)
	if err := lb.ffn.WriteInputFP16(0, lb.ffnIn); err != nil {
		return fmt.Errorf("run layer backward ffn: write input: %w", err)
	}
	if err := evalKernelTracked(lb.metrics, lb.ffn); err != nil {
		return fmt.Errorf("run layer backward ffn: eval: %w", err)
	}
	if err := lb.ffn.ReadOutputFP16(0, lb.ffnOut); err != nil {
		return fmt.Errorf("run layer backward ffn: read output: %w", err)
	}
	copy(dxNorm, lb.ffnOut[:dimN])
	copy(dh1, lb.ffnOut[dimN:dimN+hiddenN])
	return nil
}

func (lb *layerBackward) runAttention(dxNorm, dq, dk, dv, q, k, v, dx2 []float32) error {
	if lb != nil && lb.dynamic {
		return lb.runDynamicAttention(dxNorm, dq, dk, dv, q, k, v, dx2)
	}
	if lb == nil || lb.sdpa1 == nil || lb.sdpa2 == nil || lb.qkv == nil {
		return fmt.Errorf("run layer backward attention: layer is closed")
	}
	dimN := lb.dim * lb.seq
	if err := checkLen("run layer backward attention", "q", q, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward attention", "k", k, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward attention", "v", v, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward attention", "dx2", dx2, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward attention", "dq", dq, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward attention", "dk", dk, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward attention", "dv", dv, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward attention", "dx", dxNorm, dimN); err != nil {
		return err
	}

	concatInto(lb.sdpa1In, q, k, v, dx2)
	if err := lb.sdpa1.WriteInputFP16(0, lb.sdpa1In); err != nil {
		return fmt.Errorf("run layer backward attention: write sdpa1 input: %w", err)
	}
	if err := evalKernelTracked(lb.metrics, lb.sdpa1); err != nil {
		return fmt.Errorf("run layer backward attention: eval sdpa1: %w", err)
	}
	if err := model.CopyOutputChannelsToInput(lb.sdpa2, 0, 0, lb.sdpa1, 0, lb.dim, 2*lb.scoreCh); err != nil {
		return fmt.Errorf("run layer backward attention: copy sdpa1 output into sdpa2 input: %w", err)
	}
	if err := lb.sdpa2.WriteInputFP16Channels(0, 2*lb.scoreCh, q); err != nil {
		return fmt.Errorf("run layer backward attention: write q into sdpa2 input: %w", err)
	}
	if err := lb.sdpa2.WriteInputFP16Channels(0, 2*lb.scoreCh+lb.dim, k); err != nil {
		return fmt.Errorf("run layer backward attention: write k into sdpa2 input: %w", err)
	}
	if err := evalKernelTracked(lb.metrics, lb.sdpa2); err != nil {
		return fmt.Errorf("run layer backward attention: eval sdpa2: %w", err)
	}
	if err := model.CopyOutputChannelsToInput(lb.qkv, 0, 0, lb.sdpa2, 0, 0, 2*lb.dim); err != nil {
		return fmt.Errorf("run layer backward attention: copy sdpa2 output into qkv input: %w", err)
	}
	if err := model.CopyOutputChannelsToInput(lb.qkv, 0, 2*lb.dim, lb.sdpa1, 0, 0, lb.dim); err != nil {
		return fmt.Errorf("run layer backward attention: copy sdpa1 dv into qkv input: %w", err)
	}
	if err := evalKernelTracked(lb.metrics, lb.qkv); err != nil {
		return fmt.Errorf("run layer backward attention: eval qkv: %w", err)
	}
	if err := lb.sdpa2.ReadOutputFP16Channels(0, 0, dq); err != nil {
		return fmt.Errorf("run layer backward attention: read dq: %w", err)
	}
	if err := lb.sdpa2.ReadOutputFP16Channels(0, lb.dim, dk); err != nil {
		return fmt.Errorf("run layer backward attention: read dk: %w", err)
	}
	if err := lb.sdpa1.ReadOutputFP16Channels(0, 0, dv); err != nil {
		return fmt.Errorf("run layer backward attention: read dv: %w", err)
	}
	if err := lb.qkv.ReadOutputFP16(0, dxNorm); err != nil {
		return fmt.Errorf("run layer backward attention: read qkv output: %w", err)
	}
	return nil
}

func (lb *layerBackward) refreshWeights(layer stories.LayerWeights) error {
	if lb == nil {
		return fmt.Errorf("refresh layer backward: layer is nil")
	}
	if !lb.dynamic {
		return fmt.Errorf("refresh layer backward: baked weights require recompile")
	}
	return lb.stageDynamicWeights(layer)
}

func (e *Engine) ensureBackward() error {
	if e == nil {
		return fmt.Errorf("ane hybrid backward is unavailable")
	}
	if !e.hybridBackwardRequested {
		return fmt.Errorf("ane hybrid backward is disabled")
	}
	if e.backwardInit && !e.backwardDirty {
		return e.backwardInitErr
	}
	e.backwardInit = true
	e.backwardDirty = false
	if e.backwardInitErr != nil {
		return e.backwardInitErr
	}
	if !e.useANE {
		e.backwardInitErr = fmt.Errorf("ane hybrid backward is disabled")
		return e.backwardInitErr
	}
	for i := range e.backward {
		if e.backward[i] != nil {
			e.backward[i].close()
		}
	}
	backward, err := compileParallel(len(e.mw.Layers), func(i int) (*layerBackward, error) {
		lb, err := compileStoriesLayerBackwardFunc(e.mw.Layers[i], e.seq)
		if err != nil {
			return nil, fmt.Errorf("storiesane step: compile backward layer %d: %w", i, err)
		}
		return lb, nil
	}, func(lb *layerBackward) {
		if lb != nil {
			lb.close()
		}
	})
	if err != nil {
		e.backward = nil
		e.backwardInitErr = err
		return e.backwardInitErr
	}
	e.backward = backward
	return nil
}

func (e *Engine) disableHybridBackward(err error) {
	if e == nil || err == nil {
		return
	}
	for i := range e.backward {
		if e.backward[i] != nil {
			e.backward[i].close()
		}
	}
	e.backward = nil
	e.backwardInit = true
	e.backwardDirty = false
	e.backwardInitErr = err
}

func concatInto(dst []float32, parts ...[]float32) {
	off := 0
	for _, part := range parts {
		copy(dst[off:], part)
		off += len(part)
	}
}

func checkLen(op, name string, got []float32, want int) error {
	if len(got) != want {
		return fmt.Errorf("%s: %s len=%d want=%d", op, name, len(got), want)
	}
	return nil
}
