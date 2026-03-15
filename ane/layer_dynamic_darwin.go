//go:build darwin

package ane

import (
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/tmc/apple/x/ane/mil"
	"github.com/tmc/apple/x/ane/model"
	"github.com/tmc/autoresearch-go-ane/ane/stories"
	"github.com/tmc/apple/coregraphics"
	appleiosurface "github.com/tmc/apple/iosurface"
	xane "github.com/tmc/apple/x/ane"
)

var dynamicLayerSpecs struct {
	mu sync.Mutex
	m  map[int]*dynamicLayerCompileSpec
}

type dynamicLayerCompileSpec struct {
	maskBlob    []byte
	ropeCosBlob []byte
	ropeSinBlob []byte

	attMIL     string
	ffnMIL     string
	ffnW2MIL   string
	ffnTailMIL string
	wotMIL     string
	sdpa1MIL   string
	sdpa2MIL   string
	qkvMIL     string

	// Inference-only kernels (smaller output surfaces).
	attInfMIL string
	ffnInfMIL string
}

func dynamicLayerSpec(seq int) (*dynamicLayerCompileSpec, error) {
	dynamicLayerSpecs.mu.Lock()
	if dynamicLayerSpecs.m == nil {
		dynamicLayerSpecs.m = make(map[int]*dynamicLayerCompileSpec)
	}
	if spec := dynamicLayerSpecs.m[seq]; spec != nil {
		dynamicLayerSpecs.mu.Unlock()
		return spec, nil
	}
	dynamicLayerSpecs.mu.Unlock()

	const (
		dim    = stories.Dim
		hidden = stories.Hidden
		heads  = stories.Heads
	)
	maskBlob, err := mil.BuildCausalMaskBlob(seq)
	if err != nil {
		return nil, fmt.Errorf("build mask blob: %w", err)
	}
	ropeCosBlob, ropeSinBlob, err := buildRoPEBlobsLocal(seq, dim/heads)
	if err != nil {
		return nil, fmt.Errorf("build rope blobs: %w", err)
	}
	spec := &dynamicLayerCompileSpec{
		maskBlob:    maskBlob,
		ropeCosBlob: ropeCosBlob,
		ropeSinBlob: ropeSinBlob,
		attMIL:      genStoriesSDPAForwardDynamicTaps(dim, heads, seq),
		ffnMIL:      genStoriesFFNForwardDynamicTaps(dim, hidden, seq),
		ffnW2MIL:    genDynamicMatmulFP16(dim, hidden, seq),
		ffnTailMIL:  genStoriesFFNBackwardTailDynamic(dim, hidden, seq),
		wotMIL:      genDynamicMatmulFP16(dim, dim, seq),
		sdpa1MIL:    genStoriesSDPABackward1Dynamic(dim, heads, seq),
		sdpa2MIL:    mil.GenSDPABackward2(dim, heads, seq),
		qkvMIL:      genStoriesQKVBackwardDynamic(dim, heads, seq),
		attInfMIL:   genStoriesSDPAForwardDynamicInference(dim, heads, seq),
		ffnInfMIL:   genStoriesFFNForwardDynamicInference(dim, hidden, seq),
	}

	dynamicLayerSpecs.mu.Lock()
	if existing := dynamicLayerSpecs.m[seq]; existing != nil {
		dynamicLayerSpecs.mu.Unlock()
		return existing, nil
	}
	dynamicLayerSpecs.m[seq] = spec
	dynamicLayerSpecs.mu.Unlock()
	return spec, nil
}

func compileStoriesLayerForwardDynamic(layer stories.LayerWeights, seq int) (_ *layerForward, err error) {
	const (
		dim    = stories.Dim
		hidden = stories.Hidden
		heads  = stories.Heads
	)
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
	spec, err := dynamicLayerSpec(seq)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward dynamic: %w", err)
	}
	att, err := model.Compile(model.CompileOptions{
		MILText:     spec.attMIL,
		SharedModel: true,
		WeightFiles: []model.WeightFile{
			{
				Path: "@model_path/weights/mask.bin",
				Blob: spec.maskBlob,
			},
			{
				Path: "@model_path/weights/rope_cos.bin",
				Blob: spec.ropeCosBlob,
			},
			{
				Path: "@model_path/weights/rope_sin.bin",
				Blob: spec.ropeSinBlob,
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("compile layer forward dynamic: attention: %w", err)
	}
	defer func() {
		if err != nil {
			att.Close()
		}
	}()
	ffn, err := model.Compile(model.CompileOptions{
		MILText:     spec.ffnMIL,
		SharedModel: true,
	})
	if err != nil {
		return nil, fmt.Errorf("compile layer forward dynamic: ffn: %w", err)
	}
	// Compile inference-only kernels (smaller output, no taps).
	attInf, err := model.Compile(model.CompileOptions{
		MILText:     spec.attInfMIL,
		SharedModel: true,
		WeightFiles: []model.WeightFile{
			{
				Path: "@model_path/weights/mask.bin",
				Blob: spec.maskBlob,
			},
			{
				Path: "@model_path/weights/rope_cos.bin",
				Blob: spec.ropeCosBlob,
			},
			{
				Path: "@model_path/weights/rope_sin.bin",
				Blob: spec.ropeSinBlob,
			},
		},
	})
	if err != nil {
		// Non-fatal: fall back to taps kernel for inference.
		attInf = nil
	}
	ffnInf, ffnInfErr := model.Compile(model.CompileOptions{
		MILText:     spec.ffnInfMIL,
		SharedModel: true,
	})
	if ffnInfErr != nil {
		ffnInf = nil
		if attInf != nil {
			attInf.Close()
			attInf = nil
		}
	}

	lf := &layerForward{
		dim:     dim,
		hidden:  hidden,
		heads:   heads,
		seq:     seq,
		att:     att,
		ffn:     ffn,
		attInf:  attInf,
		ffnInf:  ffnInf,
		dynamic: true,
		attTaps: true,
		ffnTaps: true,
		attOut: make([]float32, 5*dim*seq),
		ffnOut:  make([]float32, (dim+hidden)*seq),
		x2:      make([]float32, dim*seq),
	}
	if err := lf.stageDynamicWeights(layerForwardWeights{
		Wq: layer.Wq,
		Wk: layer.Wk,
		Wv: layer.Wv,
		Wo: layer.Wo,
		W1: layer.W1,
		W2: layer.W2,
	}); err != nil {
		lf.close()
		return nil, err
	}
	return lf, nil
}

func compileStoriesLayerBackwardDynamic(layer stories.LayerWeights, seq int) (_ *layerBackward, err error) {
	const (
		dim    = stories.Dim
		hidden = stories.Hidden
		heads  = stories.Heads
	)
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
	spec, err := dynamicLayerSpec(seq)
	if err != nil {
		return nil, fmt.Errorf("compile layer backward dynamic: %w", err)
	}
	ffnW2, err := model.Compile(model.CompileOptions{
		MILText:     spec.ffnW2MIL,
		SharedModel: true,
	})
	if err != nil {
		return nil, fmt.Errorf("compile layer backward dynamic: ffn w2: %w", err)
	}
	defer func() {
		if err != nil {
			ffnW2.Close()
		}
	}()
	ffnTail, err := model.Compile(model.CompileOptions{
		MILText:     spec.ffnTailMIL,
		SharedModel: true,
	})
	if err != nil {
		return nil, fmt.Errorf("compile layer backward dynamic: ffn tail: %w", err)
	}
	defer func() {
		if err != nil {
			ffnTail.Close()
		}
	}()
	wot, err := model.Compile(model.CompileOptions{
		MILText:     spec.wotMIL,
		SharedModel: true,
	})
	if err != nil {
		return nil, fmt.Errorf("compile layer backward dynamic: wot: %w", err)
	}
	defer func() {
		if err != nil {
			wot.Close()
		}
	}()
	sdpa1, err := model.Compile(model.CompileOptions{
		MILText:     spec.sdpa1MIL,
		SharedModel: true,
		WeightFiles: []model.WeightFile{{
			Path: "@model_path/weights/mask.bin",
			Blob: spec.maskBlob,
		}},
	})
	if err != nil {
		return nil, fmt.Errorf("compile layer backward dynamic: sdpa1: %w", err)
	}
	defer func() {
		if err != nil {
			sdpa1.Close()
		}
	}()
	sdpa2, err := model.Compile(model.CompileOptions{
		MILText:     spec.sdpa2MIL,
		SharedModel: true,
	})
	if err != nil {
		return nil, fmt.Errorf("compile layer backward dynamic: sdpa2: %w", err)
	}
	defer func() {
		if err != nil {
			sdpa2.Close()
		}
	}()
	qkv, err := model.Compile(model.CompileOptions{
		MILText:     spec.qkvMIL,
		SharedModel: true,
	})
	if err != nil {
		return nil, fmt.Errorf("compile layer backward dynamic: qkv: %w", err)
	}
	lb := &layerBackward{
		dim:     dim,
		hidden:  hidden,
		heads:   heads,
		seq:     seq,
		scoreCh: heads * seq,
		ffnW2:   ffnW2,
		ffn:     ffnTail,
		wot:     wot,
		sdpa1:   sdpa1,
		sdpa2:   sdpa2,
		qkv:     qkv,
		dynamic: true,
		ffnOut:  make([]float32, (dim+hidden)*seq),
	}
	if err := lb.stageDynamicWeights(layer); err != nil {
		lb.close()
		return nil, err
	}
	return lb, nil
}

func (lf *layerForward) stageDynamicWeights(w layerForwardWeights) error {
	if lf == nil || !lf.dynamic {
		return fmt.Errorf("stage layer forward dynamic weights: layer is not dynamic")
	}
	if err := stageStoriesAttentionForwardWeights(lf.att, lf.seq, w); err != nil {
		return fmt.Errorf("stage layer forward dynamic weights: attention: %w", err)
	}
	if err := stageStoriesFFNForwardWeights(lf.ffn, lf.seq, lf.hidden, w); err != nil {
		return fmt.Errorf("stage layer forward dynamic weights: ffn: %w", err)
	}
	// Stage weights into inference kernels (same input layout, shared weights).
	if lf.attInf != nil {
		if err := stageStoriesAttentionForwardWeights(lf.attInf, lf.seq, w); err != nil {
			return fmt.Errorf("stage layer forward dynamic weights: attention inf: %w", err)
		}
	}
	if lf.ffnInf != nil {
		if err := stageStoriesFFNForwardWeights(lf.ffnInf, lf.seq, lf.hidden, w); err != nil {
			return fmt.Errorf("stage layer forward dynamic weights: ffn inf: %w", err)
		}
	}
	return nil
}

func (lb *layerBackward) stageDynamicWeights(layer stories.LayerWeights) error {
	if lb == nil || !lb.dynamic {
		return fmt.Errorf("stage layer backward dynamic weights: layer is not dynamic")
	}
	if err := stageDynamicMatmulWeights(lb.ffnW2, lb.seq, lb.hidden, layer.W2); err != nil {
		return fmt.Errorf("stage layer backward dynamic weights: ffn w2: %w", err)
	}
	if err := stageStoriesFFNTailWeights(lb.ffn, lb.seq, lb.hidden, layer.W1); err != nil {
		return fmt.Errorf("stage layer backward dynamic weights: ffn tail: %w", err)
	}
	if err := stageDynamicMatmulWeights(lb.wot, lb.seq, lb.dim, layer.Wo); err != nil {
		return fmt.Errorf("stage layer backward dynamic weights: wot: %w", err)
	}
	if err := stageStoriesQKVBackwardWeights(lb.qkv, lb.seq, layer.Wq, layer.Wk, layer.Wv); err != nil {
		return fmt.Errorf("stage layer backward dynamic weights: qkv: %w", err)
	}
	return nil
}

func (lf *layerForward) runDynamicWithTaps(out, x []float32, cache *layerCache) error {
	if cache == nil {
		return lf.runDynamicInferenceOnly(out, x)
	}
	if lf == nil || lf.att == nil || lf.ffn == nil {
		return fmt.Errorf("run layer forward dynamic: layer is closed")
	}
	want := lf.dim * lf.seq
	if len(x) != want {
		return fmt.Errorf("run layer forward dynamic: input len=%d want=%d", len(x), want)
	}
	if len(out) != want {
		return fmt.Errorf("run layer forward dynamic: output len=%d want=%d", len(out), want)
	}
	if err := writeStoriesAttentionForwardActs(lf.att, lf.seq, x); err != nil {
		return fmt.Errorf("run layer forward dynamic: write attention input: %w", err)
	}
	if err := evalKernelTracked(lf.metrics, lf.att); err != nil {
		return fmt.Errorf("run layer forward dynamic: eval attention: %w", err)
	}
	if err := readOutputFP16ChannelsFast(lf.att, 0, 0, lf.seq, lf.attOut); err != nil {
		return fmt.Errorf("run layer forward dynamic: read attention output: %w", err)
	}
	copy(lf.x2, lf.attOut[:want])
	// Residual is already applied inside the ANE kernel (x2 = x + Wo@attn).
	if err := writeStoriesFFNForwardActs(lf.ffn, lf.seq, lf.x2); err != nil {
		return fmt.Errorf("run layer forward dynamic: write ffn input: %w", err)
	}
	if err := evalKernelTracked(lf.metrics, lf.ffn); err != nil {
		return fmt.Errorf("run layer forward dynamic: eval ffn: %w", err)
	}
	if err := readOutputFP16ChannelsFast(lf.ffn, 0, 0, lf.seq, lf.ffnOut); err != nil {
		return fmt.Errorf("run layer forward dynamic: read ffn output: %w", err)
	}
	// FFN residual: out = x2 + ffnOutput.
	for i := 0; i < want; i++ {
		out[i] = lf.x2[i] + lf.ffnOut[i]
	}
	copy(cache.x2, lf.x2)
	cache.attTapsReady = false
	cache.ffnTapsReady = false
	rmsNormNoWeightCF(cache.xNorm, x, lf.dim, lf.seq)
	copy(cache.q, lf.attOut[want:2*want])
	copy(cache.k, lf.attOut[2*want:3*want])
	copy(cache.v, lf.attOut[3*want:4*want])
	copy(cache.attOut, lf.attOut[4*want:5*want])
	cache.attTapsReady = true
	hiddenSpan := lf.hidden * lf.seq
	copy(cache.h1, lf.ffnOut[want:want+hiddenSpan])
	for i := range cache.gate {
		cache.gate[i] = reluSquared32(cache.h1[i])
	}
	rmsNormNoWeightCF(cache.x2Norm, cache.x2, lf.dim, lf.seq)
	cache.ffnTapsReady = true
	return nil
}

// runDynamicInferenceOnly runs a dynamic layer using inference-only kernels
// when available, falling back to reading only dim channels from taps kernels.
// Inference kernels output only dim channels (no taps), shrinking IOSurfaces.
func (lf *layerForward) runDynamicInferenceOnly(out, x []float32) error {
	if lf == nil || lf.att == nil || lf.ffn == nil {
		return fmt.Errorf("run layer forward dynamic: layer is closed")
	}
	want := lf.dim * lf.seq
	if len(x) != want {
		return fmt.Errorf("run layer forward dynamic: input len=%d want=%d", len(x), want)
	}
	if len(out) != want {
		return fmt.Errorf("run layer forward dynamic: output len=%d want=%d", len(out), want)
	}

	// Use inference kernels if available, otherwise fall back to taps kernels.
	attK := lf.att
	ffnK := lf.ffn
	if lf.attInf != nil && lf.ffnInf != nil {
		attK = lf.attInf
		ffnK = lf.ffnInf
	}

	if err := writeStoriesAttentionForwardActs(attK, lf.seq, x); err != nil {
		return fmt.Errorf("run layer forward dynamic: write attention input: %w", err)
	}
	if err := evalKernelTracked(lf.metrics, attK); err != nil {
		return fmt.Errorf("run layer forward dynamic: eval attention: %w", err)
	}
	// Read only x2 (dim channels).
	if err := readOutputFP16ChannelsFast(attK, 0, 0, lf.seq, lf.x2); err != nil {
		return fmt.Errorf("run layer forward dynamic: read attention output: %w", err)
	}
	if err := writeStoriesFFNForwardActs(ffnK, lf.seq, lf.x2); err != nil {
		return fmt.Errorf("run layer forward dynamic: write ffn input: %w", err)
	}
	if err := evalKernelTracked(lf.metrics, ffnK); err != nil {
		return fmt.Errorf("run layer forward dynamic: eval ffn: %w", err)
	}
	// Read only x_next (dim channels).
	if err := readOutputFP16ChannelsFast(ffnK, 0, 0, lf.seq, out); err != nil {
		return fmt.Errorf("run layer forward dynamic: read ffn output: %w", err)
	}
	return nil
}

func (lb *layerBackward) runDynamicFFN(dxNorm, dh1, dFFN, h1 []float32) error {
	if lb == nil || lb.ffnW2 == nil || lb.ffn == nil {
		return fmt.Errorf("run layer backward dynamic ffn: layer is closed")
	}
	dimN := lb.dim * lb.seq
	hiddenN := lb.hidden * lb.seq
	if err := checkLen("run layer backward dynamic ffn", "dffn", dFFN, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward dynamic ffn", "h1", h1, hiddenN); err != nil {
		return err
	}
	if err := checkLen("run layer backward dynamic ffn", "dx", dxNorm, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward dynamic ffn", "dh1", dh1, hiddenN); err != nil {
		return err
	}
	if err := writeDynamicMatmulActs(lb.ffnW2, lb.seq, dFFN); err != nil {
		return fmt.Errorf("run layer backward dynamic ffn: write w2 input: %w", err)
	}
	if err := evalKernelTracked(lb.metrics, lb.ffnW2); err != nil {
		return fmt.Errorf("run layer backward dynamic ffn: eval w2: %w", err)
	}
	if err := model.CopyOutputRangeToInput(lb.ffn, 0, 0, 0, lb.ffnW2, 0, 0, 0, lb.hidden, lb.seq); err != nil {
		return fmt.Errorf("run layer backward dynamic ffn: copy dsilu to tail: %w", err)
	}
	if err := writeStoriesFFNTailAuxActs(lb.ffn, lb.seq, lb.hidden, h1); err != nil {
		return fmt.Errorf("run layer backward dynamic ffn: write tail aux input: %w", err)
	}
	if err := evalKernelTracked(lb.metrics, lb.ffn); err != nil {
		return fmt.Errorf("run layer backward dynamic ffn: eval tail: %w", err)
	}
	if err := readOutputFP16ChannelsFast(lb.ffn, 0, 0, lb.seq, lb.ffnOut); err != nil {
		return fmt.Errorf("run layer backward dynamic ffn: read tail output: %w", err)
	}
	copy(dxNorm, lb.ffnOut[:dimN])
	copy(dh1, lb.ffnOut[dimN:dimN+hiddenN])
	return nil
}

func (lb *layerBackward) runDynamicAttention(dxNorm, dq, dk, dv, q, k, v, dx2 []float32) error {
	if lb == nil || lb.wot == nil || lb.sdpa1 == nil || lb.sdpa2 == nil || lb.qkv == nil {
		return fmt.Errorf("run layer backward dynamic attention: layer is closed")
	}
	dimN := lb.dim * lb.seq
	if err := checkLen("run layer backward dynamic attention", "q", q, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward dynamic attention", "k", k, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward dynamic attention", "v", v, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward dynamic attention", "dx2", dx2, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward dynamic attention", "dq", dq, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward dynamic attention", "dk", dk, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward dynamic attention", "dv", dv, dimN); err != nil {
		return err
	}
	if err := checkLen("run layer backward dynamic attention", "dx", dxNorm, dimN); err != nil {
		return err
	}
	if err := writeDynamicMatmulActs(lb.wot, lb.seq, dx2); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: write wot input: %w", err)
	}
	if err := evalKernelTracked(lb.metrics, lb.wot); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: eval wot: %w", err)
	}
	if err := writeInputFP16ChannelsFast(lb.sdpa1, 0, 0, lb.seq, q); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: write q: %w", err)
	}
	if err := writeInputFP16ChannelsFast(lb.sdpa1, 0, lb.dim, lb.seq, k); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: write k: %w", err)
	}
	if err := writeInputFP16ChannelsFast(lb.sdpa1, 0, 2*lb.dim, lb.seq, v); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: write v: %w", err)
	}
	if err := model.CopyOutputChannelsToInput(lb.sdpa1, 0, 3*lb.dim, lb.wot, 0, 0, lb.dim); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: copy da: %w", err)
	}
	if err := evalKernelTracked(lb.metrics, lb.sdpa1); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: eval sdpa1: %w", err)
	}
	if err := model.CopyOutputChannelsToInput(lb.sdpa2, 0, 0, lb.sdpa1, 0, lb.dim, 2*lb.scoreCh); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: copy sdpa1 output: %w", err)
	}
	if err := writeInputFP16ChannelsFast(lb.sdpa2, 0, 2*lb.scoreCh, lb.seq, q); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: write q into sdpa2: %w", err)
	}
	if err := writeInputFP16ChannelsFast(lb.sdpa2, 0, 2*lb.scoreCh+lb.dim, lb.seq, k); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: write k into sdpa2: %w", err)
	}
	if err := evalKernelTracked(lb.metrics, lb.sdpa2); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: eval sdpa2: %w", err)
	}
	if err := model.CopyOutputRangeToInput(lb.qkv, 0, 0, 0, lb.sdpa2, 0, 0, 0, lb.dim, lb.seq); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: copy dq to qkv: %w", err)
	}
	if err := model.CopyOutputRangeToInput(lb.qkv, 0, 0, lb.seq, lb.sdpa2, 0, lb.dim, 0, lb.dim, lb.seq); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: copy dk to qkv: %w", err)
	}
	if err := model.CopyOutputRangeToInput(lb.qkv, 0, 0, 2*lb.seq, lb.sdpa1, 0, 0, 0, lb.dim, lb.seq); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: copy dv to qkv: %w", err)
	}
	if err := readOutputFP16ChannelsFast(lb.sdpa2, 0, 0, lb.seq, dq); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: read dq: %w", err)
	}
	if err := readOutputFP16ChannelsFast(lb.sdpa2, 0, lb.dim, lb.seq, dk); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: read dk: %w", err)
	}
	if err := readOutputFP16ChannelsFast(lb.sdpa1, 0, 0, lb.seq, dv); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: read dv: %w", err)
	}
	if err := evalKernelTracked(lb.metrics, lb.qkv); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: eval qkv: %w", err)
	}
	if err := readOutputFP16ChannelsFast(lb.qkv, 0, 0, lb.seq, dxNorm); err != nil {
		return fmt.Errorf("run layer backward dynamic attention: read qkv output: %w", err)
	}
	return nil
}

func stageStoriesAttentionForwardWeights(k *model.Kernel, seq int, w layerForwardWeights) error {
	width := seq + 1 + 4*stories.Dim
	return withLockedFP16Input(k, 0, func(layout xane.TensorLayout, data []uint16) error {
		if err := requireFP16InputLayout("stage stories attention forward weights", layout, stories.Dim, width); err != nil {
			return err
		}
		for d := 0; d < stories.Dim; d++ {
			row := inputRowFP16(data, layout, d)
			row[seq] = xane.Float32ToFP16(1.0) // Parameterless RMSNorm: weight = 1.0
		}
		writeTransposedMatrixFP16(data, layout, 0, seq+1, stories.Dim, stories.Dim, w.Wq)
		writeTransposedMatrixFP16(data, layout, 0, seq+1+stories.Dim, stories.Dim, stories.Dim, w.Wk)
		writeTransposedMatrixFP16(data, layout, 0, seq+1+2*stories.Dim, stories.Dim, stories.Dim, w.Wv)
		writeTransposedMatrixFP16(data, layout, 0, seq+1+3*stories.Dim, stories.Dim, stories.Dim, w.Wo)
		return nil
	})
}

func writeStoriesAttentionForwardActs(k *model.Kernel, seq int, x []float32) error {
	return withLockedFP16Input(k, 0, func(layout xane.TensorLayout, data []uint16) error {
		if err := requireFP16InputLayout("write stories attention forward acts", layout, stories.Dim, seq+1+4*stories.Dim); err != nil {
			return err
		}
		writeChannelFirstActsFP16(data, layout, seq, x)
		return nil
	})
}

func stageStoriesFFNForwardWeights(k *model.Kernel, seq, hidden int, w layerForwardWeights) error {
	width := seq + 1 + 2*hidden
	return withLockedFP16Input(k, 0, func(layout xane.TensorLayout, data []uint16) error {
		if err := requireFP16InputLayout("stage stories ffn forward weights", layout, stories.Dim, width); err != nil {
			return err
		}
		for d := 0; d < stories.Dim; d++ {
			row := inputRowFP16(data, layout, d)
			row[seq] = xane.Float32ToFP16(1.0) // Parameterless RMSNorm: weight = 1.0
		}
		writeTransposedMatrixFP16(data, layout, 0, seq+1, hidden, stories.Dim, w.W1)
		writeMatrixRowsFP16(data, layout, 0, seq+1+hidden, stories.Dim, hidden, w.W2)
		return nil
	})
}

func writeStoriesFFNForwardActs(k *model.Kernel, seq int, x []float32) error {
	return withLockedFP16Input(k, 0, func(layout xane.TensorLayout, data []uint16) error {
		if err := requireFP16InputLayout("write stories ffn forward acts", layout, stories.Dim, seq+1+2*stories.Hidden); err != nil {
			return err
		}
		writeChannelFirstActsFP16(data, layout, seq, x)
		return nil
	})
}

func stageStoriesFFNTailWeights(k *model.Kernel, seq, hidden int, w1 []float32) error {
	width := 2*seq + stories.Dim
	return withLockedFP16Input(k, 0, func(layout xane.TensorLayout, data []uint16) error {
		if err := requireFP16InputLayout("stage stories ffn tail weights", layout, hidden, width); err != nil {
			return err
		}
		writeMatrixRowsFP16(data, layout, 0, 2*seq, hidden, stories.Dim, w1)
		return nil
	})
}

func writeStoriesFFNTailAuxActs(k *model.Kernel, seq, hidden int, h1 []float32) error {
	return withLockedFP16Input(k, 0, func(layout xane.TensorLayout, data []uint16) error {
		if err := requireFP16InputLayout("write stories ffn tail acts", layout, hidden, 2*seq+stories.Dim); err != nil {
			return err
		}
		writeChannelFirstActsOffsetFP16(data, layout, 0, seq, seq, h1)
		return nil
	})
}

func stageDynamicMatmulWeights(k *model.Kernel, seq, outDim int, w []float32) error {
	inDim := len(w) / outDim
	width := seq + outDim
	return withLockedFP16Input(k, 0, func(layout xane.TensorLayout, data []uint16) error {
		if err := requireFP16InputLayout("stage dynamic matmul weights", layout, inDim, width); err != nil {
			return err
		}
		writeMatrixRowsFP16(data, layout, 0, seq, inDim, outDim, w)
		return nil
	})
}

func writeDynamicMatmulActs(k *model.Kernel, seq int, x []float32) error {
	inDim := len(x) / seq
	return withLockedFP16Input(k, 0, func(layout xane.TensorLayout, data []uint16) error {
		if err := requireFP16InputLayout("write dynamic matmul acts", layout, inDim, layout.Width); err != nil {
			return err
		}
		writeChannelFirstActsFP16(data, layout, seq, x)
		return nil
	})
}

func stageStoriesQKVBackwardWeights(k *model.Kernel, seq int, wq, wk, wv []float32) error {
	width := 3*seq + 3*stories.Dim
	return withLockedFP16Input(k, 0, func(layout xane.TensorLayout, data []uint16) error {
		if err := requireFP16InputLayout("stage stories qkv backward weights", layout, stories.Dim, width); err != nil {
			return err
		}
		writeMatrixRowsFP16(data, layout, 0, 3*seq, stories.Dim, stories.Dim, wq)
		writeMatrixRowsFP16(data, layout, 0, 3*seq+stories.Dim, stories.Dim, stories.Dim, wk)
		writeMatrixRowsFP16(data, layout, 0, 3*seq+2*stories.Dim, stories.Dim, stories.Dim, wv)
		return nil
	})
}

func withLockedFP16Input(k *model.Kernel, input int, fn func(layout xane.TensorLayout, data []uint16) error) error {
	if k == nil {
		return fmt.Errorf("kernel is nil")
	}
	layout := k.InputLayout(input)
	ref := k.InputSurface(input)
	if ref == 0 {
		return fmt.Errorf("input surface %d is nil", input)
	}
	surf := appleiosurface.IOSurfaceRef(ref)
	appleiosurface.IOSurfaceLock(surf, 0, nil)
	defer appleiosurface.IOSurfaceUnlock(surf, 0, nil)
	base := appleiosurface.IOSurfaceGetBaseAddress(surf)
	if base == nil {
		return fmt.Errorf("nil IOSurface base address")
	}
	data := unsafe.Slice((*uint16)(base), layout.AllocSize()/2)
	return fn(layout, data)
}

func withLockedFP16Output(k *model.Kernel, output int, fn func(layout xane.TensorLayout, data []uint16) error) error {
	if k == nil {
		return fmt.Errorf("kernel is nil")
	}
	layout := k.OutputLayout(output)
	ref := k.OutputSurface(output)
	if ref == 0 {
		return fmt.Errorf("output surface %d is nil", output)
	}
	surf := appleiosurface.IOSurfaceRef(ref)
	appleiosurface.IOSurfaceLock(surf, appleiosurface.KIOSurfaceLockReadOnly, nil)
	defer appleiosurface.IOSurfaceUnlock(surf, appleiosurface.KIOSurfaceLockReadOnly, nil)
	base := appleiosurface.IOSurfaceGetBaseAddress(surf)
	if base == nil {
		return fmt.Errorf("nil IOSurface base address")
	}
	data := unsafe.Slice((*uint16)(base), layout.AllocSize()/2)
	return fn(layout, data)
}

func writeInputFP16ChannelsFast(k *model.Kernel, input, channelOffset, width int, x []float32) error {
	return withLockedFP16Input(k, input, func(layout xane.TensorLayout, data []uint16) error {
		if err := requireFP16InputLayout("write input fp16 channels", layout, 0, 0); err != nil {
			return err
		}
		if width <= 0 {
			return fmt.Errorf("write input fp16 channels: invalid width=%d", width)
		}
		if len(x)%width != 0 {
			return fmt.Errorf("write input fp16 channels: len=%d width=%d", len(x), width)
		}
		channels := len(x) / width
		if channelOffset < 0 || channelOffset+channels > layout.Channels {
			return fmt.Errorf("write input fp16 channels: channel range [%d,%d) out of [0,%d)", channelOffset, channelOffset+channels, layout.Channels)
		}
		if width > layout.Width {
			return fmt.Errorf("write input fp16 channels: width=%d > layout width=%d", width, layout.Width)
		}
		writeChannelFirstActsOffsetFP16(data, layout, channelOffset, 0, width, x)
		return nil
	})
}

func readOutputFP16ChannelsFast(k *model.Kernel, output, channelOffset, width int, dst []float32) error {
	return withLockedFP16Output(k, output, func(layout xane.TensorLayout, data []uint16) error {
		if layout.Height != 1 || layout.ElemSize != 2 {
			return fmt.Errorf("read output fp16 channels: unsupported layout height=%d elem=%d", layout.Height, layout.ElemSize)
		}
		if width <= 0 {
			return fmt.Errorf("read output fp16 channels: invalid width=%d", width)
		}
		if len(dst)%width != 0 {
			return fmt.Errorf("read output fp16 channels: len=%d width=%d", len(dst), width)
		}
		channels := len(dst) / width
		if channelOffset < 0 || channelOffset+channels > layout.Channels {
			return fmt.Errorf("read output fp16 channels: channel range [%d,%d) out of [0,%d)", channelOffset, channelOffset+channels, layout.Channels)
		}
		if width > layout.Width {
			return fmt.Errorf("read output fp16 channels: width=%d > layout width=%d", width, layout.Width)
		}
		readChannelFirstActsOffsetFP16(dst, data, layout, channelOffset, 0, width)
		return nil
	})
}

func requireFP16InputLayout(op string, layout xane.TensorLayout, channels, width int) error {
	if layout.Height != 1 || layout.ElemSize != 2 {
		return fmt.Errorf("%s: unsupported layout height=%d elem=%d", op, layout.Height, layout.ElemSize)
	}
	if channels > 0 && layout.Channels != channels {
		return fmt.Errorf("%s: channels=%d want=%d", op, layout.Channels, channels)
	}
	if width > 0 && layout.Width != width {
		return fmt.Errorf("%s: width=%d want=%d", op, layout.Width, width)
	}
	return nil
}

func inputRowFP16(data []uint16, layout xane.TensorLayout, channel int) []uint16 {
	off := (channel * layout.PlaneStride) / 2
	return data[off : off+layout.Width]
}

func writeChannelFirstActsFP16(data []uint16, layout xane.TensorLayout, seq int, x []float32) {
	writeChannelFirstActsOffsetFP16(data, layout, 0, 0, seq, x)
}

func writeContiguousFP16(dst []uint16, src []float32) {
	if writeMatrixRowsOffsetFP16Fast(dst, xane.TensorLayout{PlaneStride: 2 * len(dst)}, 0, 0, 1, len(src), src) {
		return
	}
	for i, v := range src {
		dst[i] = xane.Float32ToFP16(v)
	}
}

func writeMatrixColumnFP16(dst []uint16, mat []float32, rows, cols, col int) {
	for r := 0; r < rows; r++ {
		dst[r] = xane.Float32ToFP16(mat[r*cols+col])
	}
}

func writeMatrixRowsFP16(data []uint16, layout xane.TensorLayout, channelOffset, widthOffset, rows, cols int, src []float32) {
	if writeMatrixRowsOffsetFP16Fast(data, layout, channelOffset, widthOffset, rows, cols, src) {
		return
	}
	for r := 0; r < rows; r++ {
		row := inputRowFP16(data, layout, channelOffset+r)
		writeContiguousFP16(row[widthOffset:widthOffset+cols], src[r*cols:(r+1)*cols])
	}
}

func writeTransposedMatrixFP16(data []uint16, layout xane.TensorLayout, channelOffset, widthOffset, srcRows, srcCols int, src []float32) {
	if writeTransposedMatrixOffsetFP16Fast(data, layout, channelOffset, widthOffset, srcRows, srcCols, src) {
		return
	}
	for c := 0; c < srcCols; c++ {
		row := inputRowFP16(data, layout, channelOffset+c)
		writeMatrixColumnFP16(row[widthOffset:widthOffset+srcRows], src, srcRows, srcCols, c)
	}
}

func iosurfaceRef(ref coregraphics.IOSurfaceRef) appleiosurface.IOSurfaceRef {
	return appleiosurface.IOSurfaceRef(ref)
}

// ropeTheta is the base frequency for RoPE. Nanochat uses 100000.
const ropeTheta = 100000.0

// buildRoPEBlobsLocal builds fp16 RoPE cos/sin blobs using our local theta.
func buildRoPEBlobsLocal(seq, headDim int) ([]byte, []byte, error) {
	if seq <= 0 {
		return nil, nil, fmt.Errorf("seq=%d must be > 0", seq)
	}
	if headDim <= 0 || headDim%2 != 0 {
		return nil, nil, fmt.Errorf("headDim=%d must be even and > 0", headDim)
	}
	half := headDim / 2
	cosTbl := make([]float32, seq*headDim)
	sinTbl := make([]float32, seq*headDim)
	for pos := 0; pos < seq; pos++ {
		row := pos * headDim
		for i := 0; i < half; i++ {
			freq := float64(pos) / math.Pow(ropeTheta, float64(2*i)/float64(headDim))
			c := float32(math.Cos(freq))
			s := float32(math.Sin(freq))
			even := row + 2*i
			cosTbl[even] = c
			cosTbl[even+1] = c
			sinTbl[even] = s
			sinTbl[even+1] = s
		}
	}
	cosBlob, err := mil.BuildFP16Blob(cosTbl)
	if err != nil {
		return nil, nil, err
	}
	sinBlob, err := mil.BuildFP16Blob(sinTbl)
	if err != nil {
		return nil, nil, err
	}
	return cosBlob, sinBlob, nil
}
