package ane

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/tmc/apple/x/ane/dynamicmatmul"
	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

// Options configures a pure-Go Stories training engine over .bin weights.
type Options struct {
	ModelPath         string
	Tokens            []int32
	Seq               int
	AccumSteps        int
	LR                float32
	Seed              int64
	AdamBeta1         float32
	AdamBeta2         float32
	AdamEps           float32
	WeightDecay       float32
	GradClip          float32
	LossScale         float32
	CPUClassifierHead bool
	GradTaskLimit     int
	UseANE            bool // best-effort ANE layer forward plus final-head offload on Darwin
	HybridBackward    bool // best-effort ANE backward dx propagation on Darwin
}

// State captures resumable engine state.
type State struct {
	TokenPos     uint64
	LastLoss     float32
	CumTrainMS   float64
	CumWallMS    float64
	CumSteps     uint32
	CumBatches   uint32
	AdamT        uint32
	PendingSteps uint32
}

// StepResult reports one training step.
type StepResult struct {
	Loss                   float32
	StepDuration           time.Duration
	CompileDuration        time.Duration
	StartupCompileDuration time.Duration
	ANEEvalDuration        time.Duration
	CPUWorkDuration        time.Duration
	WeightRefreshDuration  time.Duration
	FinalHeadDuration      time.Duration
	EmbedGradDuration      time.Duration
	RMSDWDuration          time.Duration
	DWGEMMDuration         time.Duration
	DWWaitDuration         time.Duration
	AdamDuration           time.Duration
}

// Engine runs the current Stories training loop.
//
// It remains a small single-stage trainer, but on Darwin it can optionally
// use ANE layer-forward kernels for full-sequence logits evaluation and
// offload the final RMSNorm, classifier, and softmax kernels.
type Engine struct {
	cfg stories.ModelConfig
	mw  *stories.ModelWeights
	opt *stories.OptimState
	off *offload

	tokens            []int32
	seq               int
	accumSteps        int
	lr                float32
	seed              int64
	adamBeta1         float32
	adamBeta2         float32
	adamEps           float32
	weightDecay       float32
	gradClip          float32
	lossScale         float32
	cpuClassifierHead bool
	useANE            bool
	offDirty          bool
	rng               uint64
	state             State
	start             time.Time

	layers                  []*layerForward
	layersInit              bool
	layersDirty             bool
	layerInitErr            error
	inferLayers             []*layerForward
	inferLayersInit         bool
	inferLayerInitErr       error
	backward                []*layerBackward
	hybridBackwardRequested bool
	backwardInit            bool
	backwardDirty           bool
	backwardInitErr         error
	tmpHidden               []float32
	caches                  []layerCache
	accum                   *modelGrad
	applyGrads              []stories.LayerWeights
	gradTasks               *gradTasks

	x             []float32
	xNorm         []float32
	logits        []float32
	dy            []float32
	dx            []float32
	gRMS          []float32
	gEmbed        []float32
	accumGRMS     []float32
	accumGEmbed   []float32
	gradGate      []float32
	gradH1        []float32
	gradH3        []float32
	gradXNorm     []float32
	gradX2        []float32
	gradAtt       []float32
	gradQ         []float32
	gradK         []float32
	gradV         []float32
	gradPrev      []float32
	ropeCos       []float32
	ropeSin       []float32
	// Pre-allocated CPU inference scratch buffers.
	cpuQF    []float32
	cpuKF    []float32
	cpuVF    []float32
	cpuAttO  []float32
	cpuX2    []float32
	cpuH1    []float32
	cpuH3    []float32
	cpuGate  []float32
	cpuFFOut []float32
	embedGradDone    chan struct{}
	asyncRefreshDone chan time.Duration // async weight refresh result
	stepMetrics      aneStepMetrics

	// KV cache state for incremental token generation.
	kvc           *kvCache
	cachedRopeCos []float32
	cachedRopeSin []float32
	cacheX        []float32
	cacheXNorm    []float32
	cacheQ        []float32
	cacheK        []float32
	cacheV        []float32
	cacheAttOut   []float32
	cacheX2       []float32
	cacheH1       []float32
	cacheH3       []float32
	cacheGate     []float32
	cacheFFOut    []float32
	cacheNext     []float32
	cacheLogits   []float32
	cacheQKV      []float32 // fused Q+K+V output [qDim + 2*kvDim]
	cacheH1H3     []float32 // fused W1+W3 output [2*hidden]
	fusedQKVW     [][]float32 // per-layer fused [qDim+2*kvDim, dim] weight matrices
	fusedW1W3     [][]float32 // per-layer fused [2*hidden, dim] weight matrices

	// FP16 inference: fused weight matrices in fp16 and scratch for fp16→fp32 conversion.
	fusedQKVFP16  [][]uint16  // per-layer fused QKV in fp16
	fusedW1W3FP16 [][]uint16  // per-layer fused W1+W3 in fp16
	fp16Scratch   []float32   // scratch buffer for largest weight matrix (fp16→fp32)

	// ANE executors for single-token KV-cached inference.
	// Compiled with batch=1 for EvalNextToken.
	aneQKV   []*dynamicmatmul.Executor // per-layer fused QKV [dim → qDim+2*kvDim]
	aneWo    []*dynamicmatmul.Executor // per-layer Wo [qDim → dim]
	aneW1W3  []*dynamicmatmul.Executor // per-layer fused W1+W3 [dim → 2*hidden]
	aneW2    []*dynamicmatmul.Executor // per-layer W2 [hidden → dim]
	aneCls   *dynamicmatmul.Executor   // classifier [dim → vocab]
	aneReady bool

	lastTokenTimings TokenTimings
}

const (
	drand48Mul  = uint64(0x5DEECE66D)
	drand48Add  = uint64(0xB)
	drand48Mask = uint64((1 << 48) - 1)
)

// Open constructs an engine with pretrained .bin weights and token data.
func Open(opts Options) (*Engine, error) {
	if opts.ModelPath == "" {
		return nil, fmt.Errorf("storiesane open: model path is empty")
	}
	seq := opts.Seq
	if seq <= 0 {
		seq = stories.SeqDefault
	}
	if len(opts.Tokens) > 0 && len(opts.Tokens) < seq+1 {
		return nil, fmt.Errorf("storiesane open: not enough tokens for seq=%d", seq)
	}
	if opts.LR <= 0 {
		opts.LR = 3e-4
	}
	if opts.AccumSteps <= 0 {
		opts.AccumSteps = 1
	}
	if opts.Seed == 0 {
		opts.Seed = 42
	}
	if opts.AdamBeta1 <= 0 {
		opts.AdamBeta1 = 0.9
	}
	if opts.AdamBeta2 <= 0 {
		opts.AdamBeta2 = 0.999
	}
	if opts.AdamEps <= 0 {
		opts.AdamEps = 1e-8
	}
	if opts.GradClip < 0 {
		opts.GradClip = 0
	}
	if opts.LossScale < 0 {
		opts.LossScale = 0
	}
	if opts.LossScale == 0 {
		opts.LossScale = 256
	}
	if opts.GradTaskLimit > 0 {
		SetGradTaskConcurrency(opts.GradTaskLimit)
	}

	// Try loading as any-config model first, fall back to legacy.
	mw, _, err := stories.LoadPretrainedAny(opts.ModelPath)
	if err != nil {
		// Fall back: try legacy load, then checkpoint.
		legacyMW := stories.NewModelWeights(stories.Vocab)
		legacyLoaded, _, legacyErr := stories.LoadPretrained(opts.ModelPath)
		if legacyErr != nil {
			legacyOpt := stories.NewOptimState(stories.Vocab)
			if _, ckptErr := stories.LoadCheckpoint(opts.ModelPath, legacyMW, legacyOpt); ckptErr != nil {
				return nil, fmt.Errorf("storiesane open: load model: %w", err)
			}
			mw = legacyMW
		} else {
			mw = legacyLoaded
		}
	}
	cfg := mw.Config

	// Compress weights to fp16 for memory-bandwidth-efficient KV-cached inference.
	// Only beneficial for large models where memory bandwidth dominates over
	// fp16→fp32 conversion overhead. Threshold: ~4GB of weight data.
	weightBytes := int64(cfg.NLayers) * int64(cfg.WqSize()+cfg.WkSize()+cfg.WvSize()+cfg.WoSize()+cfg.W1Size()+cfg.W2Size()+cfg.W3Size()) * 4
	if weightBytes > 1000*1024*1024*1024 { // fp16 conversion is slower than raw fp32 BLAS; disabled until BNNS fp16 GEMM
		mw.CompressToFP16()
	}

	var opt *stories.OptimState
	var accum *modelGrad
	if len(opts.Tokens) > 0 {
		opt = stories.NewOptimStateFromConfig(cfg)
		if opts.AccumSteps > 1 {
			accum = newModelGradFromConfig(cfg)
		}
	}

	// Re-load checkpoint with optimizer state if needed.
	if opt != nil {
		if _, ckptErr := stories.LoadCheckpoint(opts.ModelPath, mw, opt); ckptErr != nil {
			// Not a checkpoint file — that's fine, we already loaded weights above.
		}
	}

	var accumGRMS []float32
	var accumGEmbed []float32
	if accum != nil {
		accumGRMS = accum.RMSFinal
		accumGEmbed = accum.Embed
	}
	caches := make([]layerCache, cfg.NLayers)
	for i := range caches {
		caches[i] = newLayerCache(seq)
	}
	ropeCos, ropeSin := buildRoPETables(seq, cfg.HeadDim())
	applyGrads := make([]stories.LayerWeights, cfg.NLayers)
	for i := range applyGrads {
		applyGrads[i] = newLayerGradFromConfig(cfg)
	}

	dim := cfg.Dim
	qDim := cfg.QDim()
	kvDim := cfg.KVDim()
	hidden := cfg.Hidden
	vocab := cfg.Vocab

	return &Engine{
		cfg:                     cfg,
		mw:                      mw,
		opt:                     opt,
		off:                     newOffload(mw, seq, opts.UseANE, opts.CPUClassifierHead),
		tokens:                  opts.Tokens,
		seq:                     seq,
		accumSteps:              opts.AccumSteps,
		lr:                      opts.LR,
		seed:                    opts.Seed,
		adamBeta1:               opts.AdamBeta1,
		adamBeta2:               opts.AdamBeta2,
		adamEps:                 opts.AdamEps,
		weightDecay:             opts.WeightDecay,
		gradClip:                opts.GradClip,
		lossScale:               opts.LossScale,
		cpuClassifierHead:       opts.CPUClassifierHead,
		useANE:                  opts.UseANE,
		hybridBackwardRequested: opts.HybridBackward,
		rng:                     drand48Seed(opts.Seed),
		start:                   time.Now(),
		tmpHidden:               make([]float32, dim*seq),
		caches:                  caches,
		accum:                   accum,
		applyGrads:              applyGrads,
		gradTasks:               newGradTasks(),
		x:                       make([]float32, dim*seq),
		xNorm:                   make([]float32, dim*seq),
		logits:                  make([]float32, vocab*seq),
		dy:                      make([]float32, dim*seq),
		dx:                      make([]float32, dim*seq),
		gRMS:                    make([]float32, dim),
		gEmbed:                  make([]float32, vocab*dim),
		accumGRMS:               accumGRMS,
		accumGEmbed:             accumGEmbed,
		gradGate:                make([]float32, hidden*seq),
		gradH1:                  make([]float32, hidden*seq),
		gradH3:                  make([]float32, hidden*seq),
		gradXNorm:               make([]float32, dim*seq),
		gradX2:                  make([]float32, dim*seq),
		gradAtt:                 make([]float32, dim*seq),
		gradQ:                   make([]float32, dim*seq),
		gradK:                   make([]float32, dim*seq),
		gradV:                   make([]float32, dim*seq),
		gradPrev:                make([]float32, dim*seq),
		ropeCos:                 ropeCos,
		ropeSin:                 ropeSin,
		cpuQF:                   make([]float32, qDim*seq),
		cpuKF:                   make([]float32, kvDim*seq),
		cpuVF:                   make([]float32, kvDim*seq),
		cpuAttO:                 make([]float32, qDim*seq),
		cpuX2:                   make([]float32, dim*seq),
		cpuH1:                   make([]float32, hidden*seq),
		cpuH3:                   make([]float32, hidden*seq),
		cpuGate:                 make([]float32, hidden*seq),
		cpuFFOut:                make([]float32, dim*seq),
	}, nil
}

// Config returns the model architecture config.
func (e *Engine) Config() stories.ModelConfig {
	if e == nil {
		return stories.ModelConfig{}
	}
	return e.cfg
}

// LR returns the current learning rate used by optimizer updates.
func (e *Engine) LR() float32 {
	if e == nil {
		return 0
	}
	return e.lr
}

// SetLR updates the learning rate used by optimizer updates.
func (e *Engine) SetLR(lr float32) error {
	if e == nil || e.mw == nil || e.opt == nil {
		return fmt.Errorf("storiesane set lr: engine is closed")
	}
	if lr <= 0 {
		return fmt.Errorf("storiesane set lr: lr must be > 0")
	}
	e.lr = lr
	return nil
}

// Step runs one engine step.
func (e *Engine) Step() (StepResult, error) {
	if e == nil || e.mw == nil || e.opt == nil {
		return StepResult{}, fmt.Errorf("storiesane step: engine is closed")
	}
	if len(e.tokens) < e.seq+1 {
		return StepResult{}, fmt.Errorf("storiesane step: not enough tokens")
	}

	t0 := time.Now()
	e.stepMetrics.reset()
	prepareStart := time.Now()
	e.Prepare()
	startupCompile := time.Since(prepareStart)
	e.attachStepMetrics()
	limit := uint64(len(e.tokens) - e.seq - 1)
	pos := uint64(0)
	if limit > 0 {
		pos = uint64(e.nextFloat64() * float64(limit))
		if pos >= limit {
			pos = limit - 1
		}
	}
	input := e.tokens[pos : pos+uint64(e.seq)]
	target := e.tokens[pos+1 : pos+1+uint64(e.seq)]
	e.state.TokenPos = pos + uint64(e.seq)
	finalHidden, err := e.forwardTraining(input)
	if err != nil {
		return StepResult{}, err
	}
	loss, err := e.runFinalHead(finalHidden, target)
	if err != nil {
		return StepResult{}, err
	}
	compileDur := e.backwardAndUpdate(input)

	dur := time.Since(t0)
	e.state.LastLoss = loss
	e.state.CumSteps++
	e.state.CumTrainMS += float64(dur) / float64(time.Millisecond)
	e.state.CumWallMS = float64(time.Since(e.start)) / float64(time.Millisecond)
	aneDur := e.stepMetrics.aneEval()
	cpuDur := dur - startupCompile - compileDur - aneDur
	if cpuDur < 0 {
		cpuDur = 0
	}

	return StepResult{
		Loss:                   loss,
		StepDuration:           dur,
		CompileDuration:        startupCompile + compileDur,
		StartupCompileDuration: startupCompile,
		ANEEvalDuration:        aneDur,
		CPUWorkDuration:        cpuDur,
		WeightRefreshDuration:  compileDur,
		FinalHeadDuration:      e.stepMetrics.finalHead(),
		EmbedGradDuration:      e.stepMetrics.embedGrad(),
		RMSDWDuration:          e.stepMetrics.rmsDW(),
		DWGEMMDuration:         e.stepMetrics.dwGEMM(),
		DWWaitDuration:         e.stepMetrics.dwWait(),
		AdamDuration:           e.stepMetrics.adam(),
	}, nil
}

// EvalLogits evaluates one full sequence and returns per-position logits.
//
// The tokens slice must have length equal to the engine sequence length.
// It uses ANE layer kernels when available and falls back to the CPU path
// otherwise.
func (e *Engine) EvalLogits(tokens []int32) ([]float32, error) {
	if e == nil || e.mw == nil {
		return nil, fmt.Errorf("storiesane eval logits: engine is closed")
	}
	if len(tokens) != e.seq {
		return nil, fmt.Errorf("storiesane eval logits: token len=%d want=%d", len(tokens), e.seq)
	}
	out := make([]float32, len(e.logits))
	if err := e.evalLogitsInto(tokens, out); err != nil {
		return nil, err
	}
	return out, nil
}

func (e *Engine) attachStepMetrics() {
	if e == nil {
		return
	}
	if e.off != nil {
		e.off.metrics = &e.stepMetrics
	}
	for i := range e.layers {
		if e.layers[i] != nil {
			e.layers[i].metrics = &e.stepMetrics
		}
	}
	for i := range e.backward {
		if e.backward[i] != nil {
			e.backward[i].metrics = &e.stepMetrics
		}
	}
}

// Flush applies any pending accumulated gradients.
func (e *Engine) Flush() (time.Duration, error) {
	if e == nil || e.mw == nil || e.opt == nil {
		return 0, fmt.Errorf("storiesane flush: engine is closed")
	}
	e.flushPending()
	compileDur := e.waitAsyncRefresh()
	if compileDur > 0 {
		e.state.CumTrainMS += float64(compileDur) / float64(time.Millisecond)
		e.state.CumWallMS = float64(time.Since(e.start)) / float64(time.Millisecond)
	}
	return compileDur, nil
}

// State returns a copy of current engine state.
func (e *Engine) State() State {
	if e == nil {
		return State{}
	}
	return e.state
}

// LoadState restores engine state counters.
func (e *Engine) LoadState(s State) error {
	if e == nil || e.mw == nil || e.opt == nil {
		return fmt.Errorf("storiesane load state: engine is closed")
	}
	e.state = s
	return nil
}

// SaveCheckpoint persists the current model, optimizer, and training metadata.
func (e *Engine) SaveCheckpoint(path string, meta stories.TrainMeta) error {
	if e == nil || e.mw == nil || e.opt == nil {
		return fmt.Errorf("storiesane save checkpoint: engine is closed")
	}
	e.waitAsyncRefresh()
	meta.LR = e.lr
	meta.Loss = e.state.LastLoss
	meta.CumTrain = e.state.CumTrainMS
	meta.CumWall = e.state.CumWallMS
	meta.CumSteps = int(e.state.CumSteps)
	meta.CumBatches = int(e.state.CumBatches)
	meta.AdamT = int(e.state.AdamT)
	if err := stories.SaveCheckpoint(path, meta, e.mw, e.opt); err != nil {
		return err
	}
	return e.appendTrailer(path)
}

// LoadCheckpoint restores the current model, optimizer, and training metadata.
func (e *Engine) LoadCheckpoint(path string) (stories.TrainMeta, error) {
	if e == nil || e.mw == nil || e.opt == nil {
		return stories.TrainMeta{}, fmt.Errorf("storiesane load checkpoint: engine is closed")
	}
	meta, err := stories.LoadCheckpoint(path, e.mw, e.opt)
	if err != nil {
		return stories.TrainMeta{}, err
	}
	e.lr = meta.LR
	e.state = State{
		LastLoss:   meta.Loss,
		CumTrainMS: meta.CumTrain,
		CumWallMS:  meta.CumWall,
		CumSteps:   uint32(meta.CumSteps),
		CumBatches: uint32(meta.CumBatches),
		AdamT:      uint32(meta.AdamT),
	}
	if e.accum != nil {
		clearModelGrad(e.accum)
	}
	clear(e.accumGRMS)
	clear(e.accumGEmbed)
	if err := e.loadTrailer(path); err != nil {
		return stories.TrainMeta{}, err
	}
	e.rng = drand48Seed(e.seed + int64(meta.Step))
	e.start = time.Now().Add(-time.Duration(meta.CumWall * float64(time.Millisecond)))
	_ = e.refreshANERuntimeForWeights()
	return meta, nil
}

// Close releases engine resources.
func (e *Engine) Close() {
	if e == nil {
		return
	}
	e.waitAsyncRefresh()
	for i := range e.layers {
		if e.layers[i] != nil {
			e.layers[i].close()
		}
	}
	e.layers = nil
	for i := range e.inferLayers {
		if e.inferLayers[i] != nil {
			e.inferLayers[i].close()
		}
	}
	e.inferLayers = nil
	for i := range e.backward {
		if e.backward[i] != nil {
			e.backward[i].close()
		}
	}
	e.backward = nil
	if e.off != nil {
		e.off.close()
		e.off = nil
	}
	e.mw = nil
	e.opt = nil
	e.tokens = nil
	e.tmpHidden = nil
	e.caches = nil
	e.accum = nil
	e.applyGrads = nil
	e.x = nil
	e.xNorm = nil
	e.logits = nil
	e.dy = nil
	e.dx = nil
	e.gRMS = nil
	e.gEmbed = nil
	e.accumGRMS = nil
	e.accumGEmbed = nil
	e.gradGate = nil
	e.gradH1 = nil
	e.gradH3 = nil
	e.gradXNorm = nil
	e.gradX2 = nil
	e.gradAtt = nil
	e.gradQ = nil
	e.gradK = nil
	e.gradV = nil
	e.gradPrev = nil
	e.ropeCos = nil
	e.ropeSin = nil
	e.cpuQF = nil
	e.cpuKF = nil
	e.cpuVF = nil
	e.cpuAttO = nil
	e.cpuX2 = nil
	e.cpuH1 = nil
	e.cpuH3 = nil
	e.cpuGate = nil
	e.cpuFFOut = nil
	if e.gradTasks != nil {
		e.gradTasks.Close()
		e.gradTasks = nil
	}
	e.cleanupANEExecutors()
	e.kvc = nil
	e.cachedRopeCos = nil
	e.cachedRopeSin = nil
	e.cacheX = nil
	e.cacheXNorm = nil
	e.cacheQ = nil
	e.cacheK = nil
	e.cacheV = nil
	e.cacheAttOut = nil
	e.cacheX2 = nil
	e.cacheH1 = nil
	e.cacheH3 = nil
	e.cacheGate = nil
	e.cacheFFOut = nil
	e.cacheNext = nil
	e.cacheLogits = nil
	e.cacheQKV = nil
	e.cacheH1H3 = nil
	e.fusedQKVW = nil
	e.fusedW1W3 = nil
	e.fusedQKVFP16 = nil
	e.fusedW1W3FP16 = nil
	e.fp16Scratch = nil
}

func (e *Engine) flushPending() time.Duration {
	if e.accum == nil || e.state.PendingSteps == 0 {
		return 0
	}
	scale := float32(1.0 / float64(e.state.PendingSteps))
	if e.lossScale > 0 {
		scale /= e.lossScale
	}
	scaleModelGrad(e.accum, scale)
	e.clipLayerGradients(e.accum.Layers, e.accum.RMSFinal, e.accum.Embed)
	e.state.AdamT++
	adamStart := time.Now()
	t := int(e.state.AdamT)
	invBC1, invBC2 := adamBiasCorrectionInv(t, e.adamBeta1, e.adamBeta2)
	e.applyLayerAdamAll(e.accum.Layers, t, invBC1, invBC2)
	adamUpdateCFWithInv(e.mw.RMSFinal, e.accum.RMSFinal, &e.opt.RMSFinal, e.lr, e.adamBeta1, e.adamBeta2, e.adamEps, 0, invBC1, invBC2, false)
	adamUpdateCFWithInv(e.mw.Embed, e.accum.Embed, &e.opt.Embed, e.lr, e.adamBeta1, e.adamBeta2, e.adamEps, e.weightDecay, invBC1, invBC2, true)
	e.stepMetrics.addAdam(time.Since(adamStart))
	// Start kernel refresh asynchronously so the next step's data loading
	// and embedding lookup overlap with compilation.
	e.startAsyncRefresh()
	clearModelGrad(e.accum)
	e.state.PendingSteps = 0
	e.state.CumBatches++
	return 0
}

const (
	trailerMagic     = "SANE"
	trailerVersionV1 = uint32(1)
	trailerVersionV2 = uint32(2)
)

func (e *Engine) appendTrailer(path string) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		return fmt.Errorf("storiesane append trailer: %w", err)
	}
	defer f.Close()
	if _, err := f.Write([]byte(trailerMagic)); err != nil {
		return fmt.Errorf("storiesane append trailer: write magic: %w", err)
	}
	version := trailerVersionV1
	if e.accum != nil {
		version = trailerVersionV2
	}
	if err := binary.Write(f, binary.LittleEndian, version); err != nil {
		return fmt.Errorf("storiesane append trailer: write version: %w", err)
	}
	if err := binary.Write(f, binary.LittleEndian, e.state.TokenPos); err != nil {
		return fmt.Errorf("storiesane append trailer: write token pos: %w", err)
	}
	if err := binary.Write(f, binary.LittleEndian, e.state.PendingSteps); err != nil {
		return fmt.Errorf("storiesane append trailer: write pending steps: %w", err)
	}
	if version == trailerVersionV2 {
		if err := writeModelGrad(f, e.accum); err != nil {
			return fmt.Errorf("storiesane append trailer: write accum grads: %w", err)
		}
		return nil
	}
	if err := writeF32Slice(f, e.accumGRMS); err != nil {
		return fmt.Errorf("storiesane append trailer: write accum rms: %w", err)
	}
	if err := writeF32Slice(f, e.accumGEmbed); err != nil {
		return fmt.Errorf("storiesane append trailer: write accum embed: %w", err)
	}
	return nil
}

func (e *Engine) loadTrailer(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("storiesane load trailer: %w", err)
	}
	defer f.Close()
	if _, err := f.Seek(int64(storiesCheckpointSize()), io.SeekStart); err != nil {
		return fmt.Errorf("storiesane load trailer: seek: %w", err)
	}
	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return nil
		}
		return fmt.Errorf("storiesane load trailer: read magic: %w", err)
	}
	if string(magic[:]) != trailerMagic {
		return nil
	}
	var ver uint32
	if err := binary.Read(f, binary.LittleEndian, &ver); err != nil {
		return fmt.Errorf("storiesane load trailer: read version: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &e.state.TokenPos); err != nil {
		return fmt.Errorf("storiesane load trailer: read token pos: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &e.state.PendingSteps); err != nil {
		return fmt.Errorf("storiesane load trailer: read pending steps: %w", err)
	}
	switch ver {
	case trailerVersionV1:
		if err := readF32Slice(f, e.accumGRMS); err != nil {
			return fmt.Errorf("storiesane load trailer: read accum rms: %w", err)
		}
		if err := readF32Slice(f, e.accumGEmbed); err != nil {
			return fmt.Errorf("storiesane load trailer: read accum embed: %w", err)
		}
	case trailerVersionV2:
		if e.accum == nil {
			return fmt.Errorf("storiesane load trailer: trailer requires accumulation buffers")
		}
		if err := readModelGrad(f, e.accum); err != nil {
			return fmt.Errorf("storiesane load trailer: read accum grads: %w", err)
		}
	default:
		return fmt.Errorf("storiesane load trailer: unsupported version %d", ver)
	}
	return nil
}

func storiesCheckpointSize() int {
	return stories.CheckpointSize(stories.DefaultConfig())
}

func writeF32Slice(w io.Writer, vals []float32) error {
	for _, v := range vals {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	return nil
}

func readF32Slice(r io.Reader, vals []float32) error {
	for i := range vals {
		if err := binary.Read(r, binary.LittleEndian, &vals[i]); err != nil {
			return err
		}
	}
	return nil
}

func writeModelGrad(w io.Writer, g *modelGrad) error {
	for i := range g.Layers {
		layer := &g.Layers[i]
		for _, vals := range [][]float32{
			layer.Wq, layer.Wk, layer.Wv, layer.Wo,
			layer.W1, layer.W2, layer.W3,
			layer.RMSAtt, layer.RMSFFN,
		} {
			if err := writeF32Slice(w, vals); err != nil {
				return err
			}
		}
	}
	if err := writeF32Slice(w, g.RMSFinal); err != nil {
		return err
	}
	return writeF32Slice(w, g.Embed)
}

func readModelGrad(r io.Reader, g *modelGrad) error {
	for i := range g.Layers {
		layer := &g.Layers[i]
		for _, vals := range [][]float32{
			layer.Wq, layer.Wk, layer.Wv, layer.Wo,
			layer.W1, layer.W2, layer.W3,
			layer.RMSAtt, layer.RMSFFN,
		} {
			if err := readF32Slice(r, vals); err != nil {
				return err
			}
		}
	}
	if err := readF32Slice(r, g.RMSFinal); err != nil {
		return err
	}
	return readF32Slice(r, g.Embed)
}

func drand48Seed(seed int64) uint64 {
	return ((uint64(seed) & 0xffffffff) << 16) | 0x330E
}

func (e *Engine) nextFloat64() float64 {
	e.rng = (drand48Mul*e.rng + drand48Add) & drand48Mask
	return float64(e.rng) / float64(uint64(1)<<48)
}

func (e *Engine) evalLogitsInto(tokens []int32, logits []float32) error {
	if len(logits) != e.cfg.Vocab*e.seq {
		return fmt.Errorf("storiesane eval logits: logits len=%d want=%d", len(logits), e.cfg.Vocab*e.seq)
	}
	e.ensureOffload()
	if e.useANE {
		// Try inference-only layers first (skip taps layer compilation).
		if e.ensureInferLayers() == nil {
			err := e.evalLogitsANEInto(tokens, logits)
			if err == nil {
				return nil
			}
		}
		// Fallback to taps layers.
		if e.ensureLayers() == nil {
			err := e.evalLogitsANEInto(tokens, logits)
			if err == nil {
				return nil
			}
			e.disableLayerForward(err)
		}
	}
	return e.evalLogitsCPUInto(tokens, logits)
}

func (e *Engine) ensureLayers() error {
	if e.layersInit {
		return e.layerInitErr
	}
	e.layersInit = true
	if !e.useANE {
		e.layerInitErr = fmt.Errorf("ane layer forward is disabled")
		return e.layerInitErr
	}
	layers, err := compileParallel(len(e.mw.Layers), func(i int) (*layerForward, error) {
		lf, err := compileStoriesLayerForwardFunc(e.mw.Layers[i], e.seq)
		if err != nil {
			return nil, fmt.Errorf("storiesane eval logits: compile layer %d: %w", i, err)
		}
		return lf, nil
	}, func(lf *layerForward) {
		if lf != nil {
			lf.close()
		}
	})
	if err != nil {
		e.layers = nil
		e.layerInitErr = err
		return e.layerInitErr
	}
	e.layers = layers
	return nil
}

func (e *Engine) ensureInferLayers() error {
	if e.inferLayersInit {
		return e.inferLayerInitErr
	}
	e.inferLayersInit = true
	if !e.useANE {
		e.inferLayerInitErr = fmt.Errorf("ane layer forward is disabled")
		return e.inferLayerInitErr
	}
	cfg := e.cfg
	layers, err := compileParallel(len(e.mw.Layers), func(i int) (*layerForward, error) {
		// Try monolithic inference kernel first (legacy MHA path).
		lf, err := compileStoriesLayerForwardInference(e.mw.Layers[i], e.seq)
		if err == nil {
			return lf, nil
		}
		// Try GQA monolithic path (handles qDim != dim).
		lf, gqaErr := compileGQALayerForwardInference(cfg, e.mw.Layers[i], e.seq)
		if gqaErr == nil {
			return lf, nil
		}
		// Fall back to tiled compilation for large models.
		lf, tiledErr := compileStoriesLayerForwardTiled(cfg, e.mw.Layers[i], e.seq)
		if tiledErr != nil {
			return nil, fmt.Errorf("storiesane eval logits: compile infer layer %d: monolithic: %v, gqa: %v, tiled: %w", i, err, gqaErr, tiledErr)
		}
		return lf, nil
	}, func(lf *layerForward) {
		if lf != nil {
			lf.close()
		}
	})
	if err != nil {
		e.inferLayers = nil
		e.inferLayerInitErr = err
		return e.inferLayerInitErr
	}
	e.inferLayers = layers
	return nil
}

func (e *Engine) evalLogitsANEInto(tokens []int32, logits []float32) error {
	dim := e.cfg.Dim
	vocab := e.cfg.Vocab
	e.ensureOffload()
	// Try inference-only layers first (no taps, baked-in residual scale).
	useLayers := e.layers
	if e.ensureInferLayers() == nil {
		useLayers = e.inferLayers
	}
	stories.EmbedLookup(e.x, e.mw.Embed, tokens, dim, e.seq)
	cur := e.x
	next := e.tmpHidden
	for i := range useLayers {
		lf := useLayers[i]
		if lf.inferScaled && lf.dynamic && i+1 < len(useLayers) {
			if err := lf.runDynamicInferPipelined(next, cur, i, useLayers); err != nil {
				return fmt.Errorf("storiesane eval logits: layer %d: %w", i, err)
			}
		} else if err := lf.run(next, cur); err != nil {
			return fmt.Errorf("storiesane eval logits: layer %d: %w", i, err)
		}
		cur, next = next, cur
	}
	// Try direct surface copy from last FFN to RMS norm.
	lastLF := useLayers[len(useLayers)-1]
	rmsOK := false
	if e.off != nil && e.off.hasRMSForward() && lastLF != nil && lastLF.ffn != nil {
		if err := e.off.runRMSForwardFromSurface(e.xNorm, lastLF.ffn, dim, e.seq); err == nil {
			rmsOK = true
		}
	}
	if !rmsOK {
		if e.off == nil || !e.off.hasRMSForward() {
			stories.RMSNorm(e.xNorm, cur, e.mw.RMSFinal, dim, e.seq)
		} else if err := e.off.runRMSForward(e.xNorm, cur); err != nil {
			return fmt.Errorf("storiesane eval logits: final rmsnorm: %w", err)
		}
	}
	if e.off == nil || !e.off.hasClassifierForward() {
		stories.MatMulVocabSeq(logits, e.mw.Embed, e.xNorm, vocab, dim, e.seq)
	} else if err := e.off.runClassifierForward(logits, e.xNorm); err != nil {
		return fmt.Errorf("storiesane eval logits: classifier: %w", err)
	}
	return nil
}

func (e *Engine) evalLogitsCPUInto(tokens []int32, logits []float32) error {
	dim := e.cfg.Dim
	qDim := e.cfg.QDim()
	kvDim := e.cfg.KVDim()
	hidden := e.cfg.Hidden
	heads := e.cfg.Heads
	kvHeads := e.cfg.EffectiveKVHeads()
	headDim := e.cfg.HeadDim()
	vocab := e.cfg.Vocab

	x := e.x
	xNorm := e.xNorm
	next := e.tmpHidden
	qf := e.cpuQF
	kf := e.cpuKF
	vf := e.cpuVF
	attOut := e.cpuAttO
	x2 := e.cpuX2
	h1 := e.cpuH1
	h3 := e.cpuH3
	gate := e.cpuGate
	ffOut := e.cpuFFOut

	stories.EmbedLookup(x, e.mw.Embed, tokens, dim, e.seq)
	cur := x
	for i := range e.mw.Layers {
		layer := e.mw.Layers[i]
		rmsNormCF(xNorm, cur, layer.RMSAtt, dim, e.seq)
		linearCF(qf, layer.Wq, xNorm, qDim, dim, e.seq)
		linearCF(kf, layer.Wk, xNorm, kvDim, dim, e.seq)
		applyRoPECFInPlace(qf, heads, headDim, e.seq, e.ropeCos, e.ropeSin)
		applyRoPECFInPlace(kf, kvHeads, headDim, e.seq, e.ropeCos, e.ropeSin)
		linearCF(vf, layer.Wv, xNorm, kvDim, dim, e.seq)
		gqaCausalAttentionCF(attOut, qf, kf, vf, heads, kvHeads, headDim, e.seq)
		linearCF(x2, layer.Wo, attOut, dim, qDim, e.seq)
		addScaledResidual(x2, cur, x2)
		rmsNormCF(xNorm, x2, layer.RMSFFN, dim, e.seq)
		linearCF(h1, layer.W1, xNorm, hidden, dim, e.seq)
		linearCF(h3, layer.W3, xNorm, hidden, dim, e.seq)
		siluMulAccel(gate, h1, h3)
		linearCF(ffOut, layer.W2, gate, dim, hidden, e.seq)
		addScaledResidual(next, x2, ffOut)
		cur, next = next, cur
	}
	stories.RMSNorm(e.xNorm, cur, e.mw.RMSFinal, dim, e.seq)
	stories.MatMulVocabSeq(logits, e.mw.Embed, e.xNorm, vocab, dim, e.seq)
	return nil
}
