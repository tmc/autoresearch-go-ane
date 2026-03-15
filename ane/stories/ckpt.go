package stories

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const (
	ckptMagic     int32 = 0x424C5A54
	ckptVersionV2 int32 = 2
	ckptVersionV3 int32 = 3
	ckptVersionV4 int32 = 4
	ckptVersionV5 int32 = 5
)

type LayerOptimState struct {
	Wq, Wk, Wv, Wo AdamState
	W1, W2         AdamState
	VEEmbed        AdamState
	VEGate         AdamState
}

type TrainMeta struct {
	Step       int
	TotalSteps int
	LR         float32
	Loss       float32
	CumCompile float64
	CumTrain   float64
	CumWall    float64
	CumSteps   int
	CumBatches int
	AdamT      int
}

type OptimState struct {
	Layers        []LayerOptimState
	Embed         AdamState
	ResidLambdas  AdamState
	X0Lambdas     AdamState
	SmearGate     AdamState
	SmearLambda   AdamState
	BackoutLambda AdamState
}

func NewOptimState(vocab int) *OptimState {
	opt := &OptimState{
		Layers:        make([]LayerOptimState, NLayers),
		Embed:         NewAdamState(vocab * Dim),
		ResidLambdas:  NewAdamState(NLayers),
		X0Lambdas:     NewAdamState(NLayers),
		SmearGate:     NewAdamState(Dim * Dim),
		SmearLambda:   NewAdamState(1),
		BackoutLambda: NewAdamState(1),
	}
	for i := range opt.Layers {
		los := LayerOptimState{
			Wq: NewAdamState(WQSize),
			Wk: NewAdamState(WQSize),
			Wv: NewAdamState(WQSize),
			Wo: NewAdamState(WOSize),
			W1: NewAdamState(W1Size),
			W2: NewAdamState(W2Size),
		}
		if IsVELayer(i) {
			los.VEEmbed = NewAdamState(VEEmbedSize(vocab))
			los.VEGate = NewAdamState(VEGateSize())
		}
		opt.Layers[i] = los
	}
	return opt
}

func SaveCheckpoint(path string, meta TrainMeta, mw *ModelWeights, opt *OptimState) error {
	return saveCheckpoint(path, ckptVersionV5, meta, mw, opt)
}

func saveCheckpoint(path string, version int32, meta TrainMeta, mw *ModelWeights, opt *OptimState) error {
	if len(mw.Layers) != NLayers {
		return fmt.Errorf("layers=%d want=%d", len(mw.Layers), NLayers)
	}
	vocab := len(mw.Embed) / Dim
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	defer f.Close()

	writeI32 := func(v int32) error { return binary.Write(f, binary.LittleEndian, v) }
	writeF32 := func(v float32) error { return binary.Write(f, binary.LittleEndian, v) }
	writeF64 := func(v float64) error { return binary.Write(f, binary.LittleEndian, v) }

	if err := writeI32(ckptMagic); err != nil {
		return err
	}
	if err := writeI32(version); err != nil {
		return err
	}
	if err := writeI32(int32(meta.Step)); err != nil {
		return err
	}
	if err := writeI32(int32(meta.TotalSteps)); err != nil {
		return err
	}
	if err := writeI32(NLayers); err != nil {
		return err
	}
	if err := writeI32(Vocab); err != nil {
		return err
	}
	if err := writeI32(Dim); err != nil {
		return err
	}
	if err := writeI32(Hidden); err != nil {
		return err
	}
	if err := writeI32(Heads); err != nil {
		return err
	}
	if err := writeI32(SeqDefault); err != nil {
		return err
	}
	if err := writeF32(meta.LR); err != nil {
		return err
	}
	if err := writeF32(meta.Loss); err != nil {
		return err
	}
	if err := writeF64(meta.CumCompile); err != nil {
		return err
	}
	if err := writeF64(meta.CumTrain); err != nil {
		return err
	}
	if err := writeF64(meta.CumWall); err != nil {
		return err
	}
	if err := writeI32(int32(meta.CumSteps)); err != nil {
		return err
	}
	if err := writeI32(int32(meta.CumBatches)); err != nil {
		return err
	}
	if err := writeI32(int32(meta.AdamT)); err != nil {
		return err
	}
	for i := 0; i < 3; i++ {
		if err := writeI32(0); err != nil {
			return err
		}
	}

	// V5: per-layer weights (no RMS, no W3)
	for i := range mw.Layers {
		l := &mw.Layers[i]
		for _, vals := range [][]float32{l.Wq, l.Wk, l.Wv, l.Wo, l.W1, l.W2} {
			if err := writeF32s(f, vals); err != nil {
				return err
			}
		}
		if err := writeF32s(f, l.VEEmbed); err != nil {
			return err
		}
		if err := writeF32s(f, l.VEGate); err != nil {
			return err
		}
	}

	// V5: per-layer optimizer state
	for i := range mw.Layers {
		layerOpt := zeroLayerOptimState(i)
		if opt != nil && i < len(opt.Layers) {
			layerOpt = opt.Layers[i]
		}
		if err := writeLayerOptimState(f, layerOpt); err != nil {
			return err
		}
	}

	// V5: model-level weights + optim
	embedOpt := NewAdamState(len(mw.Embed))
	residLambdasOpt := NewAdamState(NLayers)
	x0LambdasOpt := NewAdamState(NLayers)
	smearGateOpt := NewAdamState(Dim * Dim)
	smearLambdaOpt := NewAdamState(1)
	backoutLambdaOpt := NewAdamState(1)
	if opt != nil {
		embedOpt = opt.Embed
		residLambdasOpt = opt.ResidLambdas
		x0LambdasOpt = opt.X0Lambdas
		smearGateOpt = opt.SmearGate
		smearLambdaOpt = opt.SmearLambda
		backoutLambdaOpt = opt.BackoutLambda
	}

	if err := writeF32s(f, mw.Embed); err != nil {
		return err
	}
	if err := writeF32s(f, embedOpt.M); err != nil {
		return err
	}
	if err := writeF32s(f, embedOpt.V); err != nil {
		return err
	}
	if err := writeF32s(f, mw.ResidLambdas); err != nil {
		return err
	}
	if err := writeF32s(f, residLambdasOpt.M); err != nil {
		return err
	}
	if err := writeF32s(f, residLambdasOpt.V); err != nil {
		return err
	}
	if err := writeF32s(f, mw.X0Lambdas); err != nil {
		return err
	}
	if err := writeF32s(f, x0LambdasOpt.M); err != nil {
		return err
	}
	if err := writeF32s(f, x0LambdasOpt.V); err != nil {
		return err
	}
	if err := writeF32s(f, mw.SmearGate); err != nil {
		return err
	}
	if err := writeF32s(f, smearGateOpt.M); err != nil {
		return err
	}
	if err := writeF32s(f, smearGateOpt.V); err != nil {
		return err
	}
	if err := writeF32s(f, mw.SmearLambda); err != nil {
		return err
	}
	if err := writeF32s(f, smearLambdaOpt.M); err != nil {
		return err
	}
	if err := writeF32s(f, smearLambdaOpt.V); err != nil {
		return err
	}
	if err := writeF32s(f, mw.BackoutLambda); err != nil {
		return err
	}
	if err := writeF32s(f, backoutLambdaOpt.M); err != nil {
		return err
	}
	if err := writeF32s(f, backoutLambdaOpt.V); err != nil {
		return err
	}

	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	_ = vocab
	return nil
}

func LoadCheckpoint(path string, mw *ModelWeights, opt *OptimState) (TrainMeta, error) {
	f, err := os.Open(path)
	if err != nil {
		return TrainMeta{}, err
	}
	defer f.Close()
	return loadCheckpointFile(f, mw, opt)
}

func loadCheckpointFile(f *os.File, mw *ModelWeights, opt *OptimState) (TrainMeta, error) {
	if opt != nil {
		zeroOptimState(opt)
	}

	readI32 := func() (int32, error) {
		var v int32
		err := binary.Read(f, binary.LittleEndian, &v)
		return v, err
	}
	readF32 := func() (float32, error) {
		var v float32
		err := binary.Read(f, binary.LittleEndian, &v)
		return v, err
	}
	readF64 := func() (float64, error) {
		var v float64
		err := binary.Read(f, binary.LittleEndian, &v)
		return v, err
	}

	magic, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	ver, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	if magic != ckptMagic || (ver != ckptVersionV2 && ver != ckptVersionV3 && ver != ckptVersionV4 && ver != ckptVersionV5) {
		return TrainMeta{}, fmt.Errorf("bad checkpoint header magic=%x version=%d", magic, ver)
	}
	step, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	totalSteps, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	nLayers, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	vocabSize, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	dim, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	hidden, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	heads, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	_, err = readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	if nLayers != NLayers || vocabSize != Vocab || dim != Dim || hidden != Hidden || heads != Heads {
		return TrainMeta{}, fmt.Errorf("checkpoint config mismatch")
	}
	lr, err := readF32()
	if err != nil {
		return TrainMeta{}, err
	}
	loss, err := readF32()
	if err != nil {
		return TrainMeta{}, err
	}
	cumCompile, err := readF64()
	if err != nil {
		return TrainMeta{}, err
	}
	cumTrain, err := readF64()
	if err != nil {
		return TrainMeta{}, err
	}
	cumWall, err := readF64()
	if err != nil {
		return TrainMeta{}, err
	}
	cumSteps, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	cumBatches, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	adamT, err := readI32()
	if err != nil {
		return TrainMeta{}, err
	}
	for i := 0; i < 3; i++ {
		if _, err := readI32(); err != nil {
			return TrainMeta{}, err
		}
	}

	meta := TrainMeta{
		Step:       int(step),
		TotalSteps: int(totalSteps),
		LR:         lr,
		Loss:       loss,
		CumCompile: cumCompile,
		CumTrain:   cumTrain,
		CumWall:    cumWall,
		CumSteps:   int(cumSteps),
		CumBatches: int(cumBatches),
		AdamT:      int(adamT),
	}

	if ver == ckptVersionV5 {
		return loadCheckpointV5(f, mw, opt, meta)
	}
	if ver == ckptVersionV4 {
		return loadCheckpointV4(f, mw, opt, meta)
	}
	return loadCheckpointV3(f, mw, opt, meta, ver)
}

func loadCheckpointV5(f *os.File, mw *ModelWeights, opt *OptimState, meta TrainMeta) (TrainMeta, error) {
	return loadCheckpointV4V5(f, mw, opt, meta, true)
}

func loadCheckpointV4(f *os.File, mw *ModelWeights, opt *OptimState, meta TrainMeta) (TrainMeta, error) {
	return loadCheckpointV4V5(f, mw, opt, meta, false)
}

// v4VEGateSize is the VEGate size used in V4 checkpoints (was Dim).
const v4VEGateSize = Dim

// v4W3Size is the W3 weight size in V4 checkpoints (Hidden*Dim).
const v4W3Size = Hidden * Dim

func loadCheckpointV4V5(f *os.File, mw *ModelWeights, opt *OptimState, meta TrainMeta, isV5 bool) (TrainMeta, error) {
	// Per-layer weights
	for i := range mw.Layers {
		l := &mw.Layers[i]
		for _, vals := range []*[]float32{&l.Wq, &l.Wk, &l.Wv, &l.Wo, &l.W1, &l.W2} {
			if err := readF32s(f, *vals); err != nil {
				return TrainMeta{}, err
			}
		}
		if !isV5 {
			// V4 had W3; skip it.
			if err := skipF32(f, v4W3Size); err != nil {
				return TrainMeta{}, err
			}
		}
		if err := readF32s(f, l.VEEmbed); err != nil {
			return TrainMeta{}, err
		}
		if isV5 {
			if err := readF32s(f, l.VEGate); err != nil {
				return TrainMeta{}, err
			}
		} else if IsVELayer(i) {
			// V4 had Dim-sized VEGate; skip it, leave VEGate zero-initialized.
			if err := skipF32(f, v4VEGateSize); err != nil {
				return TrainMeta{}, err
			}
		}
	}

	// Per-layer optim
	for i := range mw.Layers {
		if opt == nil || i >= len(opt.Layers) {
			if isV5 {
				if err := skipLayerOptimState(f, i); err != nil {
					return TrainMeta{}, err
				}
			} else {
				if err := skipLayerOptimStateV4(f, i); err != nil {
					return TrainMeta{}, err
				}
			}
			continue
		}
		if isV5 {
			if err := readLayerOptimState(f, &opt.Layers[i]); err != nil {
				return TrainMeta{}, err
			}
		} else {
			if err := readLayerOptimStateV4(f, &opt.Layers[i], i); err != nil {
				return TrainMeta{}, err
			}
		}
	}

	// Model-level: Embed + optim
	if err := readF32s(f, mw.Embed); err != nil {
		return TrainMeta{}, err
	}
	if opt != nil {
		if err := readF32s(f, opt.Embed.M); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, opt.Embed.V); err != nil {
			return TrainMeta{}, err
		}
	} else {
		if err := skipF32(f, len(mw.Embed)*2); err != nil {
			return TrainMeta{}, err
		}
	}

	// ResidLambdas + optim
	if err := readF32s(f, mw.ResidLambdas); err != nil {
		return TrainMeta{}, err
	}
	if opt != nil {
		if err := readF32s(f, opt.ResidLambdas.M); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, opt.ResidLambdas.V); err != nil {
			return TrainMeta{}, err
		}
	} else {
		if err := skipF32(f, NLayers*2); err != nil {
			return TrainMeta{}, err
		}
	}

	// X0Lambdas + optim
	if err := readF32s(f, mw.X0Lambdas); err != nil {
		return TrainMeta{}, err
	}
	if opt != nil {
		if err := readF32s(f, opt.X0Lambdas.M); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, opt.X0Lambdas.V); err != nil {
			return TrainMeta{}, err
		}
	} else {
		if err := skipF32(f, NLayers*2); err != nil {
			return TrainMeta{}, err
		}
	}

	// SmearGate + optim
	if err := readF32s(f, mw.SmearGate); err != nil {
		return TrainMeta{}, err
	}
	if opt != nil {
		if err := readF32s(f, opt.SmearGate.M); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, opt.SmearGate.V); err != nil {
			return TrainMeta{}, err
		}
	} else {
		if err := skipF32(f, Dim*Dim*2); err != nil {
			return TrainMeta{}, err
		}
	}

	// SmearLambda + optim
	if err := readF32s(f, mw.SmearLambda); err != nil {
		return TrainMeta{}, err
	}
	if opt != nil {
		if err := readF32s(f, opt.SmearLambda.M); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, opt.SmearLambda.V); err != nil {
			return TrainMeta{}, err
		}
	} else {
		if err := skipF32(f, 2); err != nil {
			return TrainMeta{}, err
		}
	}

	// BackoutLambda + optim
	if err := readF32s(f, mw.BackoutLambda); err != nil {
		return TrainMeta{}, err
	}
	if opt != nil {
		if err := readF32s(f, opt.BackoutLambda.M); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, opt.BackoutLambda.V); err != nil {
			return TrainMeta{}, err
		}
	} else {
		if err := skipF32(f, 2); err != nil {
			return TrainMeta{}, err
		}
	}

	return meta, nil
}

// readLayerOptimStateV4 reads V4 format optim state which had W3 and Dim-sized VEGate.
func readLayerOptimStateV4(r io.Reader, st *LayerOptimState, layer int) error {
	for _, vals := range [][]float32{
		st.Wq.M, st.Wq.V,
		st.Wk.M, st.Wk.V,
		st.Wv.M, st.Wv.V,
		st.Wo.M, st.Wo.V,
		st.W1.M, st.W1.V,
		st.W2.M, st.W2.V,
	} {
		if err := readF32s(r, vals); err != nil {
			return err
		}
	}
	// V4 had W3 optim; skip it.
	if err := skipF32(r, 2*v4W3Size); err != nil {
		return err
	}
	// V4 had VEEmbed optim.
	if err := readF32s(r, st.VEEmbed.M); err != nil {
		return err
	}
	if err := readF32s(r, st.VEEmbed.V); err != nil {
		return err
	}
	// V4 had Dim-sized VEGate optim; skip it.
	if IsVELayer(layer) {
		if err := skipF32(r, 2*v4VEGateSize); err != nil {
			return err
		}
	}
	return nil
}

// skipLayerOptimStateV4 skips V4 format layer optim state (has W3 and Dim-sized VEGate).
func skipLayerOptimStateV4(r io.Reader, layer int) error {
	n := 6*WQSize + 2*WOSize + 2*W1Size + 2*W2Size + 2*v4W3Size
	if IsVELayer(layer) {
		n += 2*VEEmbedSize(Vocab) + 2*v4VEGateSize
	}
	return skipF32(r, n)
}

func loadCheckpointV3(f *os.File, mw *ModelWeights, opt *OptimState, meta TrainMeta, ver int32) (TrainMeta, error) {
	// V3/V2 format has RMS weights in layers; skip them into temp buffers.
	rmsSkip := make([]float32, Dim)

	for i := range mw.Layers {
		l := &mw.Layers[i]
		if err := readF32s(f, l.Wq); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.Wk); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.Wv); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.Wo); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.W1); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.W2); err != nil {
			return TrainMeta{}, err
		}
		// V3 had W3; skip it.
		if err := skipF32(f, v4W3Size); err != nil {
			return TrainMeta{}, err
		}
		// Skip RMSAtt, RMSFFN
		if err := readF32s(f, rmsSkip); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, rmsSkip); err != nil {
			return TrainMeta{}, err
		}
		// V3 has no VE weights; VEEmbed/VEGate stay zero-initialized.
	}

	// Layer optim state: V3 has RMS optim fields; skip them.
	if ver == ckptVersionV3 {
		for i := range mw.Layers {
			if opt == nil || i >= len(opt.Layers) {
				if err := skipLayerOptimStateV3(f, i); err != nil {
					return TrainMeta{}, err
				}
				continue
			}
			if err := readLayerOptimStateV3(f, &opt.Layers[i]); err != nil {
				return TrainMeta{}, err
			}
		}
	} else {
		for i := range mw.Layers {
			if err := skipLayerOptimStateV3(f, i); err != nil {
				return TrainMeta{}, err
			}
		}
	}

	// Skip RMSFinal weight + optim
	if err := readF32s(f, rmsSkip); err != nil {
		return TrainMeta{}, err
	}
	if err := skipF32(f, Dim*2); err != nil {
		return TrainMeta{}, err
	}

	// Embed
	if err := readF32s(f, mw.Embed); err != nil {
		return TrainMeta{}, err
	}
	if opt != nil {
		if err := readF32s(f, opt.Embed.M); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, opt.Embed.V); err != nil {
			return TrainMeta{}, err
		}
	} else {
		if err := skipF32(f, len(mw.Embed)*2); err != nil {
			return TrainMeta{}, err
		}
	}

	// Initialize new fields to defaults
	initLambdasDefault(mw)

	return meta, nil
}

func zeroOptimState(opt *OptimState) {
	for i := range opt.Layers {
		clear(opt.Layers[i].Wq.M)
		clear(opt.Layers[i].Wq.V)
		clear(opt.Layers[i].Wk.M)
		clear(opt.Layers[i].Wk.V)
		clear(opt.Layers[i].Wv.M)
		clear(opt.Layers[i].Wv.V)
		clear(opt.Layers[i].Wo.M)
		clear(opt.Layers[i].Wo.V)
		clear(opt.Layers[i].W1.M)
		clear(opt.Layers[i].W1.V)
		clear(opt.Layers[i].W2.M)
		clear(opt.Layers[i].W2.V)
		clear(opt.Layers[i].VEEmbed.M)
		clear(opt.Layers[i].VEEmbed.V)
		clear(opt.Layers[i].VEGate.M)
		clear(opt.Layers[i].VEGate.V)
	}
	clear(opt.Embed.M)
	clear(opt.Embed.V)
	clear(opt.ResidLambdas.M)
	clear(opt.ResidLambdas.V)
	clear(opt.X0Lambdas.M)
	clear(opt.X0Lambdas.V)
	clear(opt.SmearGate.M)
	clear(opt.SmearGate.V)
	clear(opt.SmearLambda.M)
	clear(opt.SmearLambda.V)
	clear(opt.BackoutLambda.M)
	clear(opt.BackoutLambda.V)
}

func zeroLayerOptimState(layer int) LayerOptimState {
	los := LayerOptimState{
		Wq: NewAdamState(WQSize),
		Wk: NewAdamState(WQSize),
		Wv: NewAdamState(WQSize),
		Wo: NewAdamState(WOSize),
		W1: NewAdamState(W1Size),
		W2: NewAdamState(W2Size),
	}
	if IsVELayer(layer) {
		los.VEEmbed = NewAdamState(VEEmbedSize(Vocab))
		los.VEGate = NewAdamState(VEGateSize())
	}
	return los
}

func writeLayerOptimState(w io.Writer, st LayerOptimState) error {
	for _, vals := range [][]float32{
		st.Wq.M, st.Wq.V,
		st.Wk.M, st.Wk.V,
		st.Wv.M, st.Wv.V,
		st.Wo.M, st.Wo.V,
		st.W1.M, st.W1.V,
		st.W2.M, st.W2.V,
		st.VEEmbed.M, st.VEEmbed.V,
		st.VEGate.M, st.VEGate.V,
	} {
		if err := writeF32s(w, vals); err != nil {
			return err
		}
	}
	return nil
}

func writeZeroLayerOptimState(w io.Writer, layer int) error {
	sizes := []int{
		WQSize, WQSize,
		WQSize, WQSize,
		WQSize, WQSize,
		WOSize, WOSize,
		W1Size, W1Size,
		W2Size, W2Size,
	}
	if IsVELayer(layer) {
		veEmbedN := VEEmbedSize(Vocab)
		veGateN := VEGateSize()
		sizes = append(sizes, veEmbedN, veEmbedN, veGateN, veGateN)
	}
	for _, n := range sizes {
		if err := writeZerosF32(w, n); err != nil {
			return err
		}
	}
	return nil
}

func readLayerOptimState(r io.Reader, st *LayerOptimState) error {
	for _, vals := range [][]float32{
		st.Wq.M, st.Wq.V,
		st.Wk.M, st.Wk.V,
		st.Wv.M, st.Wv.V,
		st.Wo.M, st.Wo.V,
		st.W1.M, st.W1.V,
		st.W2.M, st.W2.V,
		st.VEEmbed.M, st.VEEmbed.V,
		st.VEGate.M, st.VEGate.V,
	} {
		if err := readF32s(r, vals); err != nil {
			return err
		}
	}
	return nil
}

// readLayerOptimStateV3 reads V3-format layer optim state which includes
// RMSAtt and RMSFFN Adam states. Those are skipped.
func readLayerOptimStateV3(r io.Reader, st *LayerOptimState) error {
	for _, vals := range [][]float32{
		st.Wq.M, st.Wq.V,
		st.Wk.M, st.Wk.V,
		st.Wv.M, st.Wv.V,
		st.Wo.M, st.Wo.V,
		st.W1.M, st.W1.V,
		st.W2.M, st.W2.V,
	} {
		if err := readF32s(r, vals); err != nil {
			return err
		}
	}
	// V3 had W3 optim; skip it.
	if err := skipF32(r, 2*v4W3Size); err != nil {
		return err
	}
	// Skip RMSAtt and RMSFFN optim (4*Dim floats)
	if err := skipF32(r, 4*Dim); err != nil {
		return err
	}
	// V3 has no VE optim state; VEEmbed/VEGate Adam states stay zero-initialized.
	return nil
}

func skipLayerOptimState(r io.Reader, layer int) error {
	n := 6*WQSize + 2*WOSize + 2*W1Size + 2*W2Size
	if IsVELayer(layer) {
		n += 2*VEEmbedSize(Vocab) + 2*VEGateSize()
	}
	return skipF32(r, n)
}

// skipLayerOptimStateV3 skips V3-format layer optim (includes W3, RMS Adam, no VE).
func skipLayerOptimStateV3(r io.Reader, _ int) error {
	return skipF32(r, 6*WQSize+2*WOSize+2*W1Size+2*W2Size+2*v4W3Size+4*Dim)
}

func writeF32s(w io.Writer, vals []float32) error {
	for i := range vals {
		if err := binary.Write(w, binary.LittleEndian, vals[i]); err != nil {
			return err
		}
	}
	return nil
}

func writeZerosF32(w io.Writer, n int) error {
	zeros := make([]byte, 4096)
	total := int64(n * 4)
	for total > 0 {
		chunk := int64(len(zeros))
		if total < chunk {
			chunk = total
		}
		if _, err := w.Write(zeros[:chunk]); err != nil {
			return err
		}
		total -= chunk
	}
	return nil
}

func skipF32(r io.Reader, n int) error {
	_, err := io.CopyN(io.Discard, r, int64(n*4))
	return err
}
