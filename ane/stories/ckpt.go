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
)

type LayerOptimState struct {
	Wq, Wk, Wv, Wo AdamState
	W1, W2, W3     AdamState
	RMSAtt, RMSFFN AdamState
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
	Layers   []LayerOptimState
	RMSFinal AdamState
	Embed    AdamState
}

func NewOptimState(vocab int) *OptimState {
	opt := &OptimState{
		Layers:   make([]LayerOptimState, NLayers),
		RMSFinal: NewAdamState(Dim),
		Embed:    NewAdamState(vocab * Dim),
	}
	for i := range opt.Layers {
		opt.Layers[i] = LayerOptimState{
			Wq:     NewAdamState(WQSize),
			Wk:     NewAdamState(WQSize),
			Wv:     NewAdamState(WQSize),
			Wo:     NewAdamState(WOSize),
			W1:     NewAdamState(W1Size),
			W2:     NewAdamState(W2Size),
			W3:     NewAdamState(W3Size),
			RMSAtt: NewAdamState(Dim),
			RMSFFN: NewAdamState(Dim),
		}
	}
	return opt
}

func SaveCheckpoint(path string, meta TrainMeta, mw *ModelWeights, opt *OptimState) error {
	return saveCheckpoint(path, ckptVersionV3, meta, mw, opt)
}

func SaveCheckpointV2(path string, meta TrainMeta, mw *ModelWeights, opt *OptimState) error {
	return saveCheckpoint(path, ckptVersionV2, meta, mw, opt)
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

	for i := range mw.Layers {
		l := &mw.Layers[i]
		if err := writeF32s(f, l.Wq); err != nil {
			return err
		}
		if err := writeF32s(f, l.Wk); err != nil {
			return err
		}
		if err := writeF32s(f, l.Wv); err != nil {
			return err
		}
		if err := writeF32s(f, l.Wo); err != nil {
			return err
		}
		if err := writeF32s(f, l.W1); err != nil {
			return err
		}
		if err := writeF32s(f, l.W2); err != nil {
			return err
		}
		if err := writeF32s(f, l.W3); err != nil {
			return err
		}
		if err := writeF32s(f, l.RMSAtt); err != nil {
			return err
		}
		if err := writeF32s(f, l.RMSFFN); err != nil {
			return err
		}
	}
	if version == ckptVersionV3 {
		for i := range mw.Layers {
			layerOpt := zeroLayerOptimState()
			if opt != nil && i < len(opt.Layers) {
				layerOpt = opt.Layers[i]
			}
			if err := writeLayerOptimState(f, layerOpt); err != nil {
				return err
			}
		}
	} else {
		for range mw.Layers {
			if err := writeZeroLayerOptimState(f); err != nil {
				return err
			}
		}
	}

	if err := writeF32s(f, mw.RMSFinal); err != nil {
		return err
	}
	finalOpt := NewAdamState(Dim)
	embedOpt := NewAdamState(len(mw.Embed))
	if opt != nil {
		finalOpt = opt.RMSFinal
		embedOpt = opt.Embed
	}
	if err := writeF32s(f, finalOpt.M); err != nil {
		return err
	}
	if err := writeF32s(f, finalOpt.V); err != nil {
		return err
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

func LoadCheckpointV2(path string, mw *ModelWeights, opt *OptimState) (TrainMeta, error) {
	return LoadCheckpoint(path, mw, opt)
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
	if magic != ckptMagic || (ver != ckptVersionV2 && ver != ckptVersionV3) {
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
		if err := readF32s(f, l.W3); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.RMSAtt); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, l.RMSFFN); err != nil {
			return TrainMeta{}, err
		}
	}
	if ver == ckptVersionV3 {
		for i := range mw.Layers {
			if opt == nil || i >= len(opt.Layers) {
				if err := skipLayerOptimState(f); err != nil {
					return TrainMeta{}, err
				}
				continue
			}
			if err := readLayerOptimState(f, &opt.Layers[i]); err != nil {
				return TrainMeta{}, err
			}
		}
	} else {
		for range mw.Layers {
			if err := skipLayerOptimState(f); err != nil {
				return TrainMeta{}, err
			}
		}
	}
	if err := readF32s(f, mw.RMSFinal); err != nil {
		return TrainMeta{}, err
	}
	if opt == nil {
		if err := skipF32(f, Dim*2); err != nil {
			return TrainMeta{}, err
		}
	} else {
		if err := readF32s(f, opt.RMSFinal.M); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, opt.RMSFinal.V); err != nil {
			return TrainMeta{}, err
		}
	}
	if err := readF32s(f, mw.Embed); err != nil {
		return TrainMeta{}, err
	}
	if opt == nil {
		if err := skipF32(f, len(mw.Embed)*2); err != nil {
			return TrainMeta{}, err
		}
	} else {
		if err := readF32s(f, opt.Embed.M); err != nil {
			return TrainMeta{}, err
		}
		if err := readF32s(f, opt.Embed.V); err != nil {
			return TrainMeta{}, err
		}
	}

	return TrainMeta{
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
	}, nil
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
		clear(opt.Layers[i].W3.M)
		clear(opt.Layers[i].W3.V)
		clear(opt.Layers[i].RMSAtt.M)
		clear(opt.Layers[i].RMSAtt.V)
		clear(opt.Layers[i].RMSFFN.M)
		clear(opt.Layers[i].RMSFFN.V)
	}
	clear(opt.RMSFinal.M)
	clear(opt.RMSFinal.V)
	clear(opt.Embed.M)
	clear(opt.Embed.V)
}

func zeroLayerOptimState() LayerOptimState {
	return LayerOptimState{
		Wq:     NewAdamState(WQSize),
		Wk:     NewAdamState(WQSize),
		Wv:     NewAdamState(WQSize),
		Wo:     NewAdamState(WOSize),
		W1:     NewAdamState(W1Size),
		W2:     NewAdamState(W2Size),
		W3:     NewAdamState(W3Size),
		RMSAtt: NewAdamState(Dim),
		RMSFFN: NewAdamState(Dim),
	}
}

func writeLayerOptimState(w io.Writer, st LayerOptimState) error {
	for _, vals := range [][]float32{
		st.Wq.M, st.Wq.V,
		st.Wk.M, st.Wk.V,
		st.Wv.M, st.Wv.V,
		st.Wo.M, st.Wo.V,
		st.W1.M, st.W1.V,
		st.W2.M, st.W2.V,
		st.W3.M, st.W3.V,
		st.RMSAtt.M, st.RMSAtt.V,
		st.RMSFFN.M, st.RMSFFN.V,
	} {
		if err := writeF32s(w, vals); err != nil {
			return err
		}
	}
	return nil
}

func writeZeroLayerOptimState(w io.Writer) error {
	for _, n := range []int{
		WQSize, WQSize,
		WQSize, WQSize,
		WQSize, WQSize,
		WOSize, WOSize,
		W1Size, W1Size,
		W2Size, W2Size,
		W3Size, W3Size,
		Dim, Dim,
		Dim, Dim,
	} {
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
		st.W3.M, st.W3.V,
		st.RMSAtt.M, st.RMSAtt.V,
		st.RMSFFN.M, st.RMSFFN.V,
	} {
		if err := readF32s(r, vals); err != nil {
			return err
		}
	}
	return nil
}

func skipLayerOptimState(r io.Reader) error {
	return skipF32(r, 6*WQSize+2*WOSize+2*W1Size+2*W2Size+2*W3Size+4*Dim)
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
