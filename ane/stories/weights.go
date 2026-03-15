package stories

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"unsafe"

	"github.com/tmc/autoresearch-go-ane/ane/mil"
)

type LayerWeights struct {
	Wq, Wk, Wv, Wo []float32
	W1, W2, W3     []float32
	RMSAtt, RMSFFN []float32
}

type ModelWeights struct {
	Config   ModelConfig
	Layers   []LayerWeights
	RMSFinal []float32
	Embed    []float32 // [vocab, dim] row-major
	SharedCL bool

	// FP16 compressed weights for memory-bandwidth-bound inference.
	// When non-nil, these are used instead of the fp32 Layers for matmuls.
	FP16Layers []LayerWeightsFP16
	EmbedFP16  []uint16 // [vocab*dim] fp16
}

// LayerWeightsFP16 holds per-layer weight matrices in fp16 (uint16).
// RMS norm weights remain in fp32 since they are tiny.
type LayerWeightsFP16 struct {
	Wq, Wk, Wv, Wo []uint16
	W1, W2, W3     []uint16
	RMSAtt, RMSFFN []float32 // kept in fp32 (tiny)
}

// NewModelWeights allocates weights for the legacy 110M model.
func NewModelWeights(vocab int) *ModelWeights {
	cfg := DefaultConfig()
	cfg.Vocab = vocab
	return NewModelWeightsFromConfig(cfg)
}

// NewModelWeightsFromConfig allocates weights for an arbitrary model config.
func NewModelWeightsFromConfig(cfg ModelConfig) *ModelWeights {
	mw := &ModelWeights{
		Config:   cfg,
		Layers:   make([]LayerWeights, cfg.NLayers),
		RMSFinal: make([]float32, cfg.Dim),
		Embed:    make([]float32, cfg.Vocab*cfg.Dim),
		SharedCL: true,
	}
	for i := range mw.Layers {
		mw.Layers[i] = LayerWeights{
			Wq:     make([]float32, cfg.WqSize()),
			Wk:     make([]float32, cfg.WkSize()),
			Wv:     make([]float32, cfg.WvSize()),
			Wo:     make([]float32, cfg.WoSize()),
			W1:     make([]float32, cfg.W1Size()),
			W2:     make([]float32, cfg.W2Size()),
			W3:     make([]float32, cfg.W3Size()),
			RMSAtt: make([]float32, cfg.Dim),
			RMSFFN: make([]float32, cfg.Dim),
		}
	}
	return mw
}

// LoadPretrainedAny loads a .bin pretrained model with any architecture.
// Returns the model weights with the Config field populated from the file header.
//
// For models where head_dim != dim/heads (e.g., Qwen3), the file size is used
// to detect the actual head_dim after reading the 7-field Llama2Config header.
func LoadPretrainedAny(path string) (*ModelWeights, ModelConfig, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, ModelConfig{}, err
	}
	defer f.Close()

	var hdr Llama2Config
	if err := binary.Read(f, binary.LittleEndian, &hdr); err != nil {
		return nil, ModelConfig{}, fmt.Errorf("read config: %w", err)
	}
	cfg := ConfigFromLlama2(hdr)

	// Detect explicit head_dim by checking file size.
	// The Llama2 header assumes head_dim = dim/heads, but models like Qwen3
	// have an explicit head_dim (e.g., 128) that differs from dim/heads (e.g., 64).
	// We detect this by comparing the expected file size against the actual size.
	fi, err := f.Stat()
	if err != nil {
		return nil, cfg, fmt.Errorf("stat model file: %w", err)
	}
	fileSize := fi.Size()
	expectedSize := pretrainedFileSize(cfg)
	if fileSize != expectedSize && cfg.Heads > 0 {
		// Try doubling head_dim (common pattern: Qwen3 uses 2x head_dim).
		cfg2 := cfg
		cfg2.HeadDimOvr = (cfg.Dim / cfg.Heads) * 2
		if pretrainedFileSize(cfg2) == fileSize {
			cfg = cfg2
		} else {
			// Brute-force search for head_dim that matches file size.
			for hd := cfg.Dim / cfg.Heads; hd <= cfg.Dim; hd++ {
				cfg2.HeadDimOvr = hd
				if pretrainedFileSize(cfg2) == fileSize {
					cfg = cfg2
					break
				}
			}
		}
	}

	mw := NewModelWeightsFromConfig(cfg)
	mw.SharedCL = hdr.VocabSize > 0

	if err := readPretrainedWeights(f, mw); err != nil {
		return nil, cfg, err
	}
	return mw, cfg, nil
}

// pretrainedFileSize returns the expected .bin file size for a given config.
func pretrainedFileSize(cfg ModelConfig) int64 {
	const headerBytes = 7 * 4 // Llama2Config: 7 x int32
	perLayer := cfg.WqSize() + cfg.WkSize() + cfg.WvSize() + cfg.WoSize() +
		cfg.W1Size() + cfg.W2Size() + cfg.W3Size() + cfg.Dim*2
	total := headerBytes + (cfg.Vocab*cfg.Dim + cfg.NLayers*perLayer + cfg.Dim) * 4
	return int64(total)
}

// readPretrainedWeights reads weight data from a .bin file into pre-allocated ModelWeights.
func readPretrainedWeights(f *os.File, mw *ModelWeights) error {
	if err := readF32s(f, mw.Embed); err != nil {
		return fmt.Errorf("read embed: %w", err)
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].RMSAtt); err != nil {
			return fmt.Errorf("read rms_att[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].Wq); err != nil {
			return fmt.Errorf("read wq[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].Wk); err != nil {
			return fmt.Errorf("read wk[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].Wv); err != nil {
			return fmt.Errorf("read wv[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].Wo); err != nil {
			return fmt.Errorf("read wo[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].RMSFFN); err != nil {
			return fmt.Errorf("read rms_ffn[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].W1); err != nil {
			return fmt.Errorf("read w1[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].W2); err != nil {
			return fmt.Errorf("read w2[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].W3); err != nil {
			return fmt.Errorf("read w3[%d]: %w", i, err)
		}
	}
	if err := readF32s(f, mw.RMSFinal); err != nil {
		return fmt.Errorf("read rms_final: %w", err)
	}
	return nil
}

// LoadPretrained loads a .bin pretrained model, validating it matches the legacy 110M config.
func LoadPretrained(path string) (*ModelWeights, Llama2Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Llama2Config{}, err
	}
	defer f.Close()

	var cfg Llama2Config
	if err := binary.Read(f, binary.LittleEndian, &cfg); err != nil {
		return nil, Llama2Config{}, fmt.Errorf("read config: %w", err)
	}
	if int(cfg.Dim) != Dim || int(cfg.HiddenDim) != Hidden || int(cfg.NLayers) != NLayers {
		return nil, cfg, fmt.Errorf("config mismatch: got dim=%d hidden=%d layers=%d", cfg.Dim, cfg.HiddenDim, cfg.NLayers)
	}
	vocab := int(cfg.VocabSize)
	if vocab < 0 {
		vocab = -vocab
	}
	mw := NewModelWeights(vocab)
	mw.SharedCL = cfg.VocabSize > 0

	if err := readF32s(f, mw.Embed); err != nil {
		return nil, cfg, fmt.Errorf("read embed: %w", err)
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].RMSAtt); err != nil {
			return nil, cfg, fmt.Errorf("read rms_att[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].Wq); err != nil {
			return nil, cfg, fmt.Errorf("read wq[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].Wk); err != nil {
			return nil, cfg, fmt.Errorf("read wk[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].Wv); err != nil {
			return nil, cfg, fmt.Errorf("read wv[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].Wo); err != nil {
			return nil, cfg, fmt.Errorf("read wo[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].RMSFFN); err != nil {
			return nil, cfg, fmt.Errorf("read rms_ffn[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].W1); err != nil {
			return nil, cfg, fmt.Errorf("read w1[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].W2); err != nil {
			return nil, cfg, fmt.Errorf("read w2[%d]: %w", i, err)
		}
	}
	for i := range mw.Layers {
		if err := readF32s(f, mw.Layers[i].W3); err != nil {
			return nil, cfg, fmt.Errorf("read w3[%d]: %w", i, err)
		}
	}
	if err := readF32s(f, mw.RMSFinal); err != nil {
		return nil, cfg, fmt.Errorf("read rms_final: %w", err)
	}
	return mw, cfg, nil
}

func RandomInit(mw *ModelWeights, seed int64) {
	cfg := mw.Config
	r := rand.New(rand.NewSource(seed))
	scaleD := float32(1.0 / math.Sqrt(float64(cfg.Dim)))
	scaleH := float32(1.0 / math.Sqrt(float64(cfg.Hidden)))
	for i := range mw.Layers {
		for j := range mw.Layers[i].Wq {
			mw.Layers[i].Wq[j] = scaleD * (2*float32(r.Float64()) - 1)
		}
		for j := range mw.Layers[i].Wk {
			mw.Layers[i].Wk[j] = scaleD * (2*float32(r.Float64()) - 1)
		}
		for j := range mw.Layers[i].Wv {
			mw.Layers[i].Wv[j] = scaleD * (2*float32(r.Float64()) - 1)
		}
		for j := range mw.Layers[i].Wo {
			mw.Layers[i].Wo[j] = scaleD * (2*float32(r.Float64()) - 1)
		}
		for j := range mw.Layers[i].W1 {
			mw.Layers[i].W1[j] = scaleH * (2*float32(r.Float64()) - 1)
			mw.Layers[i].W3[j] = scaleH * (2*float32(r.Float64()) - 1)
		}
		for j := range mw.Layers[i].W2 {
			mw.Layers[i].W2[j] = scaleD * (2*float32(r.Float64()) - 1)
		}
		for j := range mw.Layers[i].RMSAtt {
			mw.Layers[i].RMSAtt[j] = 1
			mw.Layers[i].RMSFFN[j] = 1
		}
	}
	for i := range mw.RMSFinal {
		mw.RMSFinal[i] = 1
	}
	for i := range mw.Embed {
		mw.Embed[i] = 0.02 * (2*float32(r.Float64()) - 1)
	}
}

func readF32s(r io.Reader, dst []float32) error {
	if len(dst) == 0 {
		return nil
	}
	if nativeLittleEndian {
		b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(dst))), len(dst)*4)
		_, err := io.ReadFull(r, b)
		return err
	}
	buf := make([]byte, len(dst)*4)
	if _, err := io.ReadFull(r, buf); err != nil {
		return err
	}
	for i := range dst {
		off := i * 4
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[off : off+4]))
	}
	return nil
}

// CompressToFP16 populates FP16Layers and EmbedFP16 from the fp32 Layers.
// This halves memory bandwidth for the KV-cached inference path.
func (mw *ModelWeights) CompressToFP16() {
	cfg := mw.Config
	mw.FP16Layers = make([]LayerWeightsFP16, cfg.NLayers)
	for i := range mw.Layers {
		l := &mw.Layers[i]
		mw.FP16Layers[i] = LayerWeightsFP16{
			Wq:     convertSliceToFP16(l.Wq),
			Wk:     convertSliceToFP16(l.Wk),
			Wv:     convertSliceToFP16(l.Wv),
			Wo:     convertSliceToFP16(l.Wo),
			W1:     convertSliceToFP16(l.W1),
			W2:     convertSliceToFP16(l.W2),
			W3:     convertSliceToFP16(l.W3),
			RMSAtt: l.RMSAtt,
			RMSFFN: l.RMSFFN,
		}
	}
	mw.EmbedFP16 = convertSliceToFP16(mw.Embed)
}

func convertSliceToFP16(src []float32) []uint16 {
	dst := make([]uint16, len(src))
	for i, v := range src {
		dst[i] = mil.Float32ToFP16(v)
	}
	return dst
}

var nativeLittleEndian = func() bool {
	var v uint16 = 1
	return *(*byte)(unsafe.Pointer(&v)) == 1
}()
