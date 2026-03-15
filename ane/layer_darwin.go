//go:build darwin

package ane

import (
	"fmt"

	"github.com/tmc/apple/x/ane/mil"
	"github.com/tmc/apple/x/ane/model"
	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

type layerForwardWeights struct {
	Wq []float32
	Wk []float32
	Wv []float32
	Wo []float32
	W1 []float32
	W2 []float32
}

type layerForward struct {
	dim    int
	hidden int
	heads  int
	seq    int

	qkv *model.Kernel
	att *model.Kernel
	ffn *model.Kernel

	metrics  *aneStepMetrics
	dynamic  bool
	attSplit bool
	attTaps  bool
	ffnTaps  bool

	qkvOut []float32
	attOut []float32
	ffnOut []float32
	x2     []float32
}

var compileStoriesLayerForwardFunc = compileStoriesLayerForward

func compileStoriesLayerForward(layer stories.LayerWeights, seq int) (*layerForward, error) {
	if lf, err := compileStoriesLayerForwardDynamic(layer, seq); err == nil {
		return lf, nil
	}
	return compileLayerForward(stories.Dim, stories.Hidden, stories.Heads, seq, layerForwardWeights{
		Wq: layer.Wq,
		Wk: layer.Wk,
		Wv: layer.Wv,
		Wo: layer.Wo,
		W1: layer.W1,
		W2: layer.W2,
	})
}

func compileLayerForward(dim, hidden, heads, seq int, w layerForwardWeights) (_ *layerForward, err error) {
	if dim <= 0 || hidden <= 0 || heads <= 0 || seq <= 0 {
		return nil, fmt.Errorf("compile layer forward: invalid shape dim=%d hidden=%d heads=%d seq=%d", dim, hidden, heads, seq)
	}
	if dim%heads != 0 {
		return nil, fmt.Errorf("compile layer forward: dim=%d is not divisible by heads=%d", dim, heads)
	}
	if err := validateLayerWeights(dim, hidden, w); err != nil {
		return nil, err
	}
	rmsOnes := onesSlice(dim)
	rmsAttBlob, err := mil.BuildVectorWeightBlob(rmsOnes)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: rms_att blob: %w", err)
	}
	wqBlob, err := mil.BuildWeightBlob(w.Wq, dim, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: wq blob: %w", err)
	}
	wkBlob, err := mil.BuildWeightBlob(w.Wk, dim, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: wk blob: %w", err)
	}
	wvBlob, err := mil.BuildWeightBlob(w.Wv, dim, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: wv blob: %w", err)
	}
	woBlob, err := mil.BuildWeightBlob(w.Wo, dim, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: wo blob: %w", err)
	}
	maskBlob, err := mil.BuildCausalMaskBlob(seq)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: mask blob: %w", err)
	}
	rmsFFNBlob, err := mil.BuildVectorWeightBlob(rmsOnes)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: rms_ffn blob: %w", err)
	}
	w1Blob, err := mil.BuildWeightBlob(w.W1, hidden, dim)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: w1 blob: %w", err)
	}
	w2Blob, err := mil.BuildWeightBlob(w.W2, dim, hidden)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: w2 blob: %w", err)
	}
	preferCompact := seq > stories.SeqDefault
	attWeights := []model.WeightFile{
		{Path: "@model_path/weights/rms1.bin", Blob: rmsAttBlob},
		{Path: "@model_path/weights/wq.bin", Blob: wqBlob},
		{Path: "@model_path/weights/wk.bin", Blob: wkBlob},
		{Path: "@model_path/weights/wv.bin", Blob: wvBlob},
		{Path: "@model_path/weights/wo.bin", Blob: woBlob},
		{Path: "@model_path/weights/mask.bin", Blob: maskBlob},
	}
	var (
		qkv      *model.Kernel
		att      *model.Kernel
		attSplit bool
		attTaps  bool
		qkvOutCh int
		attOutCh int
	)
	if preferCompact {
		qkv, att, qkvOutCh, attOutCh, err = compileLayerForwardAttentionSplit(dim, heads, seq, attWeights)
		if err == nil {
			attSplit = true
		}
	}
	if att == nil {
		att, attTaps, attOutCh, err = compileLayerForwardAttention(dim, heads, seq, attWeights, preferCompact)
	}
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: attention: %w", err)
	}
	defer func() {
		if err != nil {
			closeKernel(qkv)
			closeKernel(att)
		}
	}()

	ffnWeights := []model.WeightFile{
		{Path: "@model_path/weights/rms2.bin", Blob: rmsFFNBlob},
		{Path: "@model_path/weights/w1.bin", Blob: w1Blob},
		{Path: "@model_path/weights/w2.bin", Blob: w2Blob},
	}
	ffn, ffnTaps, ffnOutCh, err := compileLayerForwardFFN(dim, hidden, seq, ffnWeights, preferCompact)
	if err != nil {
		return nil, fmt.Errorf("compile layer forward: ffn: %w", err)
	}

	return &layerForward{
		dim:      dim,
		hidden:   hidden,
		heads:    heads,
		seq:      seq,
		qkv:      qkv,
		att:      att,
		ffn:      ffn,
		attSplit: attSplit,
		attTaps:  attTaps,
		ffnTaps:  ffnTaps,
		qkvOut: make([]float32, qkvOutCh*seq),
		attOut:   make([]float32, attOutCh*seq),
		ffnOut:   make([]float32, ffnOutCh*seq),
		x2:       make([]float32, dim*seq),
	}, nil
}

func (lf *layerForward) close() {
	if lf == nil {
		return
	}
	closeKernel(lf.qkv)
	closeKernel(lf.att)
	closeKernel(lf.ffn)
	lf.qkv = nil
	lf.att = nil
	lf.ffn = nil
	lf.qkvOut = nil
	lf.attOut = nil
	lf.ffnOut = nil
	lf.x2 = nil
}

func (lf *layerForward) run(out, x []float32) error {
	return lf.runWithTaps(out, x, nil)
}

func (lf *layerForward) runWithTaps(out, x []float32, cache *layerCache) error {
	if lf != nil && lf.dynamic {
		return lf.runDynamicWithTaps(out, x, cache)
	}
	if lf == nil || lf.att == nil || lf.ffn == nil {
		return fmt.Errorf("run layer forward: layer is closed")
	}
	want := lf.dim * lf.seq
	if len(x) != want {
		return fmt.Errorf("run layer forward: input len=%d want=%d", len(x), want)
	}
	if len(out) != want {
		return fmt.Errorf("run layer forward: output len=%d want=%d", len(out), want)
	}

	if lf.attSplit {
		if err := lf.qkv.WriteInputFP16(0, x); err != nil {
			return fmt.Errorf("run layer forward: write qkv input: %w", err)
		}
		if err := evalKernelTracked(lf.metrics, lf.qkv); err != nil {
			return fmt.Errorf("run layer forward: eval qkv: %w", err)
		}
		if cache != nil {
			if err := lf.qkv.ReadOutputFP16(0, lf.qkvOut); err != nil {
				return fmt.Errorf("run layer forward: read qkv output: %w", err)
			}
		}
		if err := lf.att.WriteInputFP16(0, x); err != nil {
			return fmt.Errorf("run layer forward: write attention residual input: %w", err)
		}
		if err := model.CopyOutputChannelsToInput(lf.att, 1, 0, lf.qkv, 0, 0, 3*lf.dim); err != nil {
			if cache == nil {
				if err := lf.qkv.ReadOutputFP16(0, lf.qkvOut); err != nil {
					return fmt.Errorf("run layer forward: read qkv output fallback: %w", err)
				}
			}
			if err := lf.att.WriteInputFP16(1, lf.qkvOut); err != nil {
				return fmt.Errorf("run layer forward: write attention qkv input: %w", err)
			}
		}
		if err := evalKernelTracked(lf.metrics, lf.att); err != nil {
			return fmt.Errorf("run layer forward: eval attention apply: %w", err)
		}
		if err := lf.att.ReadOutputFP16(0, lf.attOut); err != nil {
			return fmt.Errorf("run layer forward: read attention apply output: %w", err)
		}
		copy(lf.x2, lf.attOut[:want])
	} else {
		if err := lf.att.WriteInputFP16(0, x); err != nil {
			return fmt.Errorf("run layer forward: write attention input: %w", err)
		}
		if err := evalKernelTracked(lf.metrics, lf.att); err != nil {
			return fmt.Errorf("run layer forward: eval attention: %w", err)
		}
		if err := lf.att.ReadOutputFP16(0, lf.attOut); err != nil {
			return fmt.Errorf("run layer forward: read attention output: %w", err)
		}
		if lf.attTaps {
			copy(lf.x2, lf.attOut[:want])
		} else {
			copy(lf.x2, lf.attOut)
		}
	}
	// Residual is already applied inside the ANE kernel (x2 = x + Wo@attn).

	if err := model.CopyOutputChannelsToInput(lf.ffn, 0, 0, lf.att, 0, 0, lf.dim); err != nil {
		if err := lf.ffn.WriteInputFP16(0, lf.x2); err != nil {
			return fmt.Errorf("run layer forward: write ffn input: %w", err)
		}
	}
	if err := evalKernelTracked(lf.metrics, lf.ffn); err != nil {
		return fmt.Errorf("run layer forward: eval ffn: %w", err)
	}
	if err := lf.ffn.ReadOutputFP16(0, lf.ffnOut); err != nil {
		return fmt.Errorf("run layer forward: read ffn output: %w", err)
	}
	ffnMain := lf.ffnOut
	if lf.ffnTaps {
		ffnMain = lf.ffnOut[:want]
	}
	// FFN residual: out = x2 + ffnOutput.
	for i := range out[:len(ffnMain)] {
		out[i] = lf.x2[i] + ffnMain[i]
	}
	if cache != nil {
		copy(cache.x2, lf.x2)
		cache.attTapsReady = false
		cache.ffnTapsReady = false
		if lf.attSplit {
			rmsNormNoWeightCF(cache.xNorm, x, lf.dim, lf.seq)
			copy(cache.q, lf.qkvOut[:want])
			copy(cache.k, lf.qkvOut[want:2*want])
			copy(cache.v, lf.qkvOut[2*want:3*want])
			copy(cache.attOut, lf.attOut[want:2*want])
			cache.attTapsReady = true
		} else if lf.attTaps {
			rmsNormNoWeightCF(cache.xNorm, x, lf.dim, lf.seq)
			copy(cache.q, lf.attOut[want:2*want])
			copy(cache.k, lf.attOut[2*want:3*want])
			copy(cache.v, lf.attOut[3*want:4*want])
			copy(cache.attOut, lf.attOut[4*want:5*want])
			cache.attTapsReady = true
		}
		if lf.ffnTaps {
			hiddenSpan := lf.hidden * lf.seq
			copy(cache.h1, lf.ffnOut[want:want+hiddenSpan])
			for i := range cache.gate {
				cache.gate[i] = reluSquared32(cache.h1[i])
			}
			rmsNormNoWeightCF(cache.x2Norm, cache.x2, lf.dim, lf.seq)
			cache.ffnTapsReady = true
		}
	}
	return nil
}

func (lf *layerForward) refreshWeights(w layerForwardWeights) error {
	if lf == nil {
		return fmt.Errorf("refresh layer forward: layer is nil")
	}
	if !lf.dynamic {
		return fmt.Errorf("refresh layer forward: baked weights require recompile")
	}
	return lf.stageDynamicWeights(w)
}

func compileLayerForwardAttentionSplit(dim, heads, seq int, weights []model.WeightFile) (*model.Kernel, *model.Kernel, int, int, error) {
	qkvWeights := make([]model.WeightFile, 0, 4)
	attWeights := make([]model.WeightFile, 0, 2)
	for _, wf := range weights {
		switch wf.Path {
		case "@model_path/weights/rms1.bin", "@model_path/weights/wq.bin", "@model_path/weights/wk.bin", "@model_path/weights/wv.bin":
			qkvWeights = append(qkvWeights, wf)
		case "@model_path/weights/wo.bin", "@model_path/weights/mask.bin":
			attWeights = append(attWeights, wf)
		}
	}
	qkv, err := compileMultiFP16Kernel(mil.GenQKVForwardRMS(dim, seq), qkvWeights, dim, 3*dim, seq)
	if err != nil {
		return nil, nil, 0, 0, fmt.Errorf("split qkv compile failed: %w", err)
	}
	att, err := compileMultiFP16Kernel(mil.GenSDPAApplyForward(dim, heads, seq), attWeights, dim, 2*dim, seq)
	if err != nil {
		closeKernel(qkv)
		return nil, nil, 0, 0, fmt.Errorf("split apply compile failed: %w", err)
	}
	return qkv, att, 3 * dim, 2 * dim, nil
}

func compileLayerForwardAttention(dim, heads, seq int, weights []model.WeightFile, preferCompact bool) (*model.Kernel, bool, int, error) {
	if !preferCompact {
		k, err := compileMultiFP16Kernel(mil.GenSDPAForwardTaps(dim, heads, seq), weights, dim, 5*dim, seq)
		if err == nil {
			return k, true, 5 * dim, nil
		}
		k, compactErr := compileMultiFP16Kernel(mil.GenSDPAForward(dim, heads, seq), weights, dim, dim, seq)
		if compactErr == nil {
			return k, false, dim, nil
		}
		return nil, false, 0, fmt.Errorf("tap compile failed: %w; compact compile failed: %v", err, compactErr)
	}
	k, err := compileMultiFP16Kernel(mil.GenSDPAForward(dim, heads, seq), weights, dim, dim, seq)
	if err != nil {
		return nil, false, 0, err
	}
	return k, false, dim, nil
}

func compileLayerForwardFFN(dim, hidden, seq int, weights []model.WeightFile, preferCompact bool) (*model.Kernel, bool, int, error) {
	if !preferCompact {
		k, err := compileMultiFP16Kernel(mil.GenFFNForwardTaps(dim, hidden, seq), weights, dim, dim+2*hidden, seq)
		if err == nil {
			return k, true, dim + 2*hidden, nil
		}
		k, compactErr := compileMultiFP16Kernel(mil.GenFFNForwardRMS(dim, hidden, seq), weights, dim, dim, seq)
		if compactErr == nil {
			return k, false, dim, nil
		}
		return nil, false, 0, fmt.Errorf("tap compile failed: %w; compact compile failed: %v", err, compactErr)
	}
	k, err := compileMultiFP16Kernel(mil.GenFFNForwardRMS(dim, hidden, seq), weights, dim, dim, seq)
	if err != nil {
		return nil, false, 0, err
	}
	return k, false, dim, nil
}

func validateLayerWeights(dim, hidden int, w layerForwardWeights) error {
	check := func(name string, got, want int) error {
		if got != want {
			return fmt.Errorf("compile layer forward: %s len=%d want=%d", name, got, want)
		}
		return nil
	}
	if err := check("wq", len(w.Wq), dim*dim); err != nil {
		return err
	}
	if err := check("wk", len(w.Wk), dim*dim); err != nil {
		return err
	}
	if err := check("wv", len(w.Wv), dim*dim); err != nil {
		return err
	}
	if err := check("wo", len(w.Wo), dim*dim); err != nil {
		return err
	}
	if err := check("w1", len(w.W1), hidden*dim); err != nil {
		return err
	}
	if err := check("w2", len(w.W2), dim*hidden); err != nil {
		return err
	}
	return nil
}

func compileMultiFP16Kernel(milText string, weights []model.WeightFile, inCh, outCh, seq int) (*model.Kernel, error) {
	return model.Compile(model.CompileOptions{
		MILText:     milText,
		WeightFiles: weights,
	})
}
