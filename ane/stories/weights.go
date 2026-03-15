package stories

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"unsafe"
)

type LayerWeights struct {
	Wq, Wk, Wv, Wo []float32
	W1, W2         []float32
	VEEmbed        []float32 // [vocab*dim] embedding table, VE layers only
	VEGate         []float32 // [Heads*VEGateChannels] per-head gate weight, VE layers only
}

// IsVELayer reports whether layer i uses value embeddings.
// Matches nanochat: alternating, last layer always included.
func IsVELayer(i int) bool { return i%2 == (NLayers-1)%2 }

// VEEmbedSize returns the number of float32s in a VE embedding table.
func VEEmbedSize(vocab int) int { return vocab * Dim }

// VEGateChannels is the number of input channels each per-head gate dot-products.
const VEGateChannels = 12

// VEGateSize returns the number of float32s in a VE gate vector.
func VEGateSize() int { return Heads * VEGateChannels }

type ModelWeights struct {
	Layers        []LayerWeights
	Embed         []float32 // [vocab, dim] row-major
	ResidLambdas  []float32 // [NLayers] per-layer residual scale
	X0Lambdas     []float32 // [NLayers] per-layer x0 scale
	SmearGate     []float32 // [Dim*Dim] linear projection for bigram gating
	SmearLambda   []float32 // [1] scalar gate strength
	BackoutLambda []float32 // [1] mid-layer residual subtraction strength
	SharedCL      bool
}

func NewModelWeights(vocab int) *ModelWeights {
	mw := &ModelWeights{
		Layers:        make([]LayerWeights, NLayers),
		Embed:         make([]float32, vocab*Dim),
		ResidLambdas:  make([]float32, NLayers),
		X0Lambdas:     make([]float32, NLayers),
		SmearGate:     make([]float32, Dim*Dim),
		SmearLambda:   make([]float32, 1),
		BackoutLambda: make([]float32, 1),
		SharedCL:      true,
	}
	for i := range mw.Layers {
		lw := LayerWeights{
			Wq: make([]float32, WQSize),
			Wk: make([]float32, WQSize),
			Wv: make([]float32, WQSize),
			Wo: make([]float32, WOSize),
			W1: make([]float32, W1Size),
			W2: make([]float32, W2Size),
		}
		if IsVELayer(i) {
			lw.VEEmbed = make([]float32, VEEmbedSize(vocab))
			lw.VEGate = make([]float32, VEGateSize())
		}
		mw.Layers[i] = lw
	}
	return mw
}

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
	// Old Llama2 format has RMSAtt weights; skip them.
	skip := make([]float32, Dim)
	for i := range mw.Layers {
		if err := readF32s(f, skip); err != nil {
			return nil, cfg, fmt.Errorf("skip rms_att[%d]: %w", i, err)
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
	// Old Llama2 format has RMSFFN weights; skip them.
	for i := range mw.Layers {
		if err := readF32s(f, skip); err != nil {
			return nil, cfg, fmt.Errorf("skip rms_ffn[%d]: %w", i, err)
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
	// Old Llama2 format has W3; skip it.
	for i := range mw.Layers {
		skip := make([]float32, W1Size)
		if err := readF32s(f, skip); err != nil {
			return nil, cfg, fmt.Errorf("skip w3[%d]: %w", i, err)
		}
	}
	// Old Llama2 format has RMSFinal weight; skip it.
	if err := readF32s(f, skip); err != nil {
		return nil, cfg, fmt.Errorf("skip rms_final: %w", err)
	}
	initLambdasDefault(mw)
	return mw, cfg, nil
}

func initLambdasDefault(mw *ModelWeights) {
	for i := range mw.ResidLambdas {
		mw.ResidLambdas[i] = 1.0
	}
	// X0Lambdas default to 0 (already zero-valued).
	mw.SmearLambda[0] = 0
	mw.BackoutLambda[0] = 0
}

func RandomInit(mw *ModelWeights, seed int64) {
	r := rand.New(rand.NewSource(seed))
	scaleD := float32(1.0 / math.Sqrt(Dim))
	for i := range mw.Layers {
		s := float32(math.Pow(3, 0.5)) * scaleD // nanochat: 3^0.5 * dim^-0.5
		for j := range mw.Layers[i].Wq {
			mw.Layers[i].Wq[j] = s * (2*float32(r.Float64()) - 1)
			mw.Layers[i].Wk[j] = s * (2*float32(r.Float64()) - 1)
			mw.Layers[i].Wv[j] = s * (2*float32(r.Float64()) - 1)
		}
		// Wo zero-init (nanochat: output projections start at zero)
		for j := range mw.Layers[i].W1 {
			mw.Layers[i].W1[j] = s * (2*float32(r.Float64()) - 1)
		}
		// W2 zero-init (nanochat: FFN output projection starts at zero)
		for j := range mw.Layers[i].VEEmbed {
			mw.Layers[i].VEEmbed[j] = s * (2*float32(r.Float64()) - 1)
		}
		// VEGate: small positive init (nanochat: uniform 0..0.02)
		for j := range mw.Layers[i].VEGate {
			mw.Layers[i].VEGate[j] = 0.02 * float32(r.Float64())
		}
	}
	for i := range mw.Embed {
		mw.Embed[i] = float32(r.NormFloat64()) // nanochat: normal * 1.0
	}
	// ResidLambdas: uniform 1.0 (nanochat default)
	for i := range mw.ResidLambdas {
		mw.ResidLambdas[i] = 1.0
	}
	// X0Lambdas: uniform 0.1 (nanochat default)
	for i := range mw.X0Lambdas {
		mw.X0Lambdas[i] = 0.1
	}
	for i := range mw.SmearGate {
		mw.SmearGate[i] = scaleD * (2*float32(r.Float64()) - 1)
	}
	mw.SmearLambda[0] = 0.01
	mw.BackoutLambda[0] = 0.01
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

var nativeLittleEndian = func() bool {
	var v uint16 = 1
	return *(*byte)(unsafe.Pointer(&v)) == 1
}()
