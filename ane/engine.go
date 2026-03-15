package ane

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

// Options configures a pure-Go Stories training engine over .bin weights.
type Options struct {
	ModelPath         string
	Tokens            []uint16
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

	// Per-param-group LR multipliers (relative to LR). Zero uses defaults.
	EmbedLRMult  float32 // embedding params (Embed, VEEmbed)
	ScalarLRMult float32 // scalar params (VEGate, SmearLambda, BackoutLambda)
	LambdaLRMult float32 // lambda params (ResidLambdas, X0Lambdas), relative to scalar LR

	// Per-param-group betas. Zero uses defaults (AdamBeta1, AdamBeta2).
	LambdaBeta1 float32 // ResidLambdas/X0Lambdas beta1
	LambdaBeta2 float32 // ResidLambdas/X0Lambdas beta2
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
	mw  *stories.ModelWeights
	opt *stories.OptimState
	off *offload

	tokens            []uint16
	seq               int
	accumSteps        int
	lr                float32
	seed              int64
	adamBeta1         float32
	adamBeta2         float32
	adamEps           float32
	weightDecay       float32
	embedLR           float32
	scalarLR          float32
	lambdaLR          float32
	lambdaBeta1       float32
	lambdaBeta2       float32
	embedLRMult       float32
	scalarLRMult      float32
	lambdaLRMult      float32
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

	x                []float32
	x0               []float32 // [dim*seq] saved post-embed-norm for residual lambdas
	xPreEmbedNorm    []float32 // [dim*seq] pre-embed-norm (for backward)
	xNorm            []float32
	logits           []float32
	logitsPreSoftcap []float32 // [vocab*seq] logits before softcap (for backward)
	dy               []float32
	dx               []float32
	gEmbed           []float32
	gResidLambdas    []float32 // [NLayers] residual lambda gradients
	gX0Lambdas       []float32 // [NLayers] x0 lambda gradients
	gX0              []float32 // [dim*seq] accumulated x0 gradient
	gradGate         []float32
	gradH1           []float32
	gradXNorm        []float32
	gradX2           []float32
	gradAtt          []float32
	gradQ            []float32
	gradK            []float32
	gradV            []float32
	gradPrev         []float32
	ropeCos          []float32
	ropeSin          []float32

	// Smear/Backout buffers.
	smearGatePre     []float32 // [dim*seq] saved gate pre-activation
	smearShifted     []float32 // [dim*seq] saved shifted input
	smearGateAct     []float32 // [dim*seq] saved sigmoid(gatePre)
	smearXPre        []float32 // [dim*seq] input before smear (for backward)
	xMid             []float32 // [dim*seq] mid-layer residual snapshot
	gSmearGate       []float32 // [dim*dim] smear gate gradient
	gSmearLambda     []float32 // [1] smear lambda gradient
	gBackoutLambda   []float32 // [1] backout lambda gradient
	accumGSmearGate  []float32 // [dim*dim] accumulated smear gate gradient
	accumGSmearLam   []float32 // [1] accumulated smear lambda gradient
	accumGBackoutLam []float32 // [1] accumulated backout lambda gradient

	// Pre-allocated CPU inference buffers (avoid per-call allocation).
	cpuQF      []float32 // [dim*seq]
	cpuKF      []float32 // [dim*seq]
	cpuVF      []float32 // [dim*seq]
	cpuAttO    []float32 // [dim*seq]
	cpuX2      []float32 // [dim*seq]
	cpuH1      []float32 // [hidden*seq]
	cpuGate    []float32 // [hidden*seq]
	cpuFFOut   []float32 // [dim*seq]
	cpuVEScr   []float32 // [dim*seq]
	cpuVEGate  []float32 // [heads*seq]
	cpuX0Inf   []float32 // [dim*seq]

	embedGradDone    chan struct{}
	asyncRefreshDone chan time.Duration // async weight refresh result
	stepMetrics      aneStepMetrics
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
	if opts.EmbedLRMult <= 0 {
		opts.EmbedLRMult = 1.0
	}
	if opts.ScalarLRMult <= 0 {
		opts.ScalarLRMult = 1.0
	}
	if opts.LambdaLRMult <= 0 {
		opts.LambdaLRMult = 0.01
	}
	if opts.LambdaBeta1 <= 0 {
		opts.LambdaBeta1 = opts.AdamBeta1
	}
	if opts.LambdaBeta2 <= 0 {
		opts.LambdaBeta2 = opts.AdamBeta2
	}
	if opts.GradTaskLimit > 0 {
		SetGradTaskConcurrency(opts.GradTaskLimit)
	}

	var opt *stories.OptimState
	var accum *modelGrad
	if len(opts.Tokens) > 0 {
		opt = stories.NewOptimState(stories.Vocab)
		if opts.AccumSteps > 1 {
			accum = newModelGrad(stories.Vocab)
		}
	}
	mw := stories.NewModelWeights(stories.Vocab)
	loaded, _, err := stories.LoadPretrained(opts.ModelPath)
	if err != nil {
		if _, ckptErr := stories.LoadCheckpoint(opts.ModelPath, mw, opt); ckptErr != nil {
			return nil, fmt.Errorf("storiesane open: load model: %w", err)
		}
	} else {
		*mw = *loaded
	}
	var accumGSmearGate []float32
	var accumGSmearLam []float32
	var accumGBackoutLam []float32
	if accum != nil {
		accumGSmearGate = accum.SmearGate
		accumGSmearLam = accum.SmearLambda
		accumGBackoutLam = accum.BackoutLambda
	}
	caches := make([]layerCache, stories.NLayers)
	for i := range caches {
		caches[i] = newLayerCache(seq, i)
	}
	ropeCos, ropeSin := buildRoPETables(seq, stories.Dim/stories.Heads)
	applyGrads := make([]stories.LayerWeights, stories.NLayers)
	for i := range applyGrads {
		applyGrads[i] = newLayerGrad(i)
	}

	return &Engine{
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
		embedLR:                 opts.LR * opts.EmbedLRMult,
		scalarLR:                opts.LR * opts.ScalarLRMult,
		lambdaLR:                opts.LR * opts.ScalarLRMult * opts.LambdaLRMult,
		lambdaBeta1:             opts.LambdaBeta1,
		lambdaBeta2:             opts.LambdaBeta2,
		embedLRMult:             opts.EmbedLRMult,
		scalarLRMult:            opts.ScalarLRMult,
		lambdaLRMult:            opts.LambdaLRMult,
		gradClip:                opts.GradClip,
		lossScale:               opts.LossScale,
		cpuClassifierHead:       opts.CPUClassifierHead,
		useANE:                  opts.UseANE,
		hybridBackwardRequested: opts.HybridBackward,
		rng:                     drand48Seed(opts.Seed),
		start:                   time.Now(),
		tmpHidden:               make([]float32, stories.Dim*seq),
		caches:                  caches,
		accum:                   accum,
		applyGrads:              applyGrads,
		gradTasks:               newGradTasks(),
		x:                       make([]float32, stories.Dim*seq),
		x0:                      make([]float32, stories.Dim*seq),
		xPreEmbedNorm:           make([]float32, stories.Dim*seq),
		xNorm:                   make([]float32, stories.Dim*seq),
		logits:                  make([]float32, stories.Vocab*seq),
		logitsPreSoftcap:        make([]float32, stories.Vocab*seq),
		dy:                      make([]float32, stories.Dim*seq),
		dx:                      make([]float32, stories.Dim*seq),
		gEmbed:                  make([]float32, len(mw.Embed)),
		gResidLambdas:           make([]float32, stories.NLayers),
		gX0Lambdas:              make([]float32, stories.NLayers),
		gX0:                     make([]float32, stories.Dim*seq),
		gradGate:                make([]float32, stories.Hidden*seq),
		gradH1:                  make([]float32, stories.Hidden*seq),
		gradXNorm:               make([]float32, stories.Dim*seq),
		gradX2:                  make([]float32, stories.Dim*seq),
		gradAtt:                 make([]float32, stories.Dim*seq),
		gradQ:                   make([]float32, stories.Dim*seq),
		gradK:                   make([]float32, stories.Dim*seq),
		gradV:                   make([]float32, stories.Dim*seq),
		gradPrev:                make([]float32, stories.Dim*seq),
		ropeCos:          ropeCos,
		ropeSin:          ropeSin,
		smearGatePre:     make([]float32, stories.Dim*seq),
		smearShifted:     make([]float32, stories.Dim*seq),
		smearGateAct:     make([]float32, stories.Dim*seq),
		smearXPre:        make([]float32, stories.Dim*seq),
		xMid:             make([]float32, stories.Dim*seq),
		gSmearGate:       make([]float32, stories.Dim*stories.Dim),
		gSmearLambda:     make([]float32, 1),
		gBackoutLambda:   make([]float32, 1),
		accumGSmearGate:  accumGSmearGate,
		accumGSmearLam:   accumGSmearLam,
		accumGBackoutLam: accumGBackoutLam,
		cpuQF:            make([]float32, stories.Dim*seq),
		cpuKF:            make([]float32, stories.Dim*seq),
		cpuVF:            make([]float32, stories.Dim*seq),
		cpuAttO:          make([]float32, stories.Dim*seq),
		cpuX2:            make([]float32, stories.Dim*seq),
		cpuH1:            make([]float32, stories.Hidden*seq),
		cpuGate:          make([]float32, stories.Hidden*seq),
		cpuFFOut:         make([]float32, stories.Dim*seq),
		cpuVEScr:         make([]float32, stories.Dim*seq),
		cpuVEGate:        make([]float32, stories.Heads*seq),
		cpuX0Inf:         make([]float32, stories.Dim*seq),
	}, nil
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
	e.embedLR = lr * e.embedLRMult
	e.scalarLR = lr * e.scalarLRMult
	e.lambdaLR = lr * e.scalarLRMult * e.lambdaLRMult
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
func (e *Engine) EvalLogits(tokens []uint16) ([]float32, error) {
	if e == nil || e.mw == nil {
		return nil, fmt.Errorf("storiesane eval logits: engine is closed")
	}
	if len(tokens) != e.seq {
		return nil, fmt.Errorf("storiesane eval logits: token len=%d want=%d", len(tokens), e.seq)
	}
	if err := e.evalLogitsInto(tokens, e.logits); err != nil {
		return nil, err
	}
	out := make([]float32, len(e.logits))
	copy(out, e.logits)
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
	clear(e.accumGSmearGate)
	clear(e.accumGSmearLam)
	clear(e.accumGBackoutLam)
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
	e.x0 = nil
	e.xPreEmbedNorm = nil
	e.xNorm = nil
	e.logits = nil
	e.logitsPreSoftcap = nil
	e.dy = nil
	e.dx = nil
	e.gEmbed = nil
	e.gResidLambdas = nil
	e.gX0Lambdas = nil
	e.gX0 = nil
	e.gradGate = nil
	e.gradH1 = nil
	e.gradXNorm = nil
	e.gradX2 = nil
	e.gradAtt = nil
	e.gradQ = nil
	e.gradK = nil
	e.gradV = nil
	e.gradPrev = nil
	e.ropeCos = nil
	e.ropeSin = nil
	e.smearGatePre = nil
	e.smearShifted = nil
	e.smearGateAct = nil
	e.smearXPre = nil
	e.xMid = nil
	e.gSmearGate = nil
	e.gSmearLambda = nil
	e.gBackoutLambda = nil
	e.accumGSmearGate = nil
	e.accumGSmearLam = nil
	e.accumGBackoutLam = nil
	e.cpuQF = nil
	e.cpuKF = nil
	e.cpuVF = nil
	e.cpuAttO = nil
	e.cpuX2 = nil
	e.cpuH1 = nil
	e.cpuGate = nil
	e.cpuFFOut = nil
	e.cpuVEScr = nil
	e.cpuVEGate = nil
	e.cpuX0Inf = nil
	if e.gradTasks != nil {
		e.gradTasks.Close()
		e.gradTasks = nil
	}
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
	e.clipLayerGradients(e.accum.Layers, e.accum.Embed)
	e.state.AdamT++
	adamStart := time.Now()
	t := int(e.state.AdamT)
	// Matrix params: base LR, base betas, weight decay.
	invBC1, invBC2 := adamBiasCorrectionInv(t, e.adamBeta1, e.adamBeta2)
	e.applyLayerAdamAll(e.accum.Layers, t, invBC1, invBC2)
	// SmearGate is a matrix param.
	adamUpdateCFWithInv(e.mw.SmearGate, e.accum.SmearGate, &e.opt.SmearGate, e.lr, e.adamBeta1, e.adamBeta2, e.adamEps, e.weightDecay, invBC1, invBC2, false)
	// Embed params: embed LR, no weight decay.
	invBC1e, invBC2e := adamBiasCorrectionInv(t, e.adamBeta1, e.adamBeta2)
	adamUpdateCFWithInv(e.mw.Embed, e.accum.Embed, &e.opt.Embed, e.embedLR, e.adamBeta1, e.adamBeta2, e.adamEps, 0, invBC1e, invBC2e, true)
	// Scalar params: scalar LR, no weight decay.
	adamUpdateCFWithInv(e.mw.SmearLambda, e.accum.SmearLambda, &e.opt.SmearLambda, e.scalarLR, e.adamBeta1, e.adamBeta2, e.adamEps, 0, invBC1, invBC2, false)
	adamUpdateCFWithInv(e.mw.BackoutLambda, e.accum.BackoutLambda, &e.opt.BackoutLambda, e.scalarLR, e.adamBeta1, e.adamBeta2, e.adamEps, 0, invBC1, invBC2, false)
	// Lambda params: lambda LR, custom betas.
	invBC1l, invBC2l := adamBiasCorrectionInv(t, e.lambdaBeta1, e.lambdaBeta2)
	adamUpdateCFWithInv(e.mw.ResidLambdas, e.accum.ResidLambdas, &e.opt.ResidLambdas, e.lambdaLR, e.lambdaBeta1, e.lambdaBeta2, e.adamEps, 0, invBC1l, invBC2l, false)
	adamUpdateCFWithInv(e.mw.X0Lambdas, e.accum.X0Lambdas, &e.opt.X0Lambdas, e.lambdaLR, e.lambdaBeta1, e.lambdaBeta2, e.adamEps, 0, invBC1l, invBC2l, false)
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
	if err := writeModelGrad(f, e.accum); err != nil {
		return fmt.Errorf("storiesane append trailer: write accum grads: %w", err)
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
		// V1 trailers stored accumGRMS and accumGEmbed which no longer exist.
		// Skip them.
		rmsBytes := int64(stories.Dim) * 4
		embedBytes := int64(stories.Vocab*stories.Dim) * 4
		if _, err := f.Seek(rmsBytes+embedBytes, io.SeekCurrent); err != nil {
			return fmt.Errorf("storiesane load trailer: skip v1 accum fields: %w", err)
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
	const headerBytes = 96
	// V5 format: per-layer weights (no RMS, no W3), VE, per-layer optim, model-level.
	layerWeights := stories.NLayers * (stories.WQSize*3 + stories.WOSize + stories.W1Size + stories.W2Size)
	layerOpt := stories.NLayers * (stories.WQSize*6 + stories.WOSize*2 + stories.W1Size*2 + stories.W2Size*2)
	veEmbedN := stories.VEEmbedSize(stories.Vocab)
	veGateN := stories.VEGateSize()
	for i := 0; i < stories.NLayers; i++ {
		if stories.IsVELayer(i) {
			layerWeights += veEmbedN + veGateN
			layerOpt += 2*veEmbedN + 2*veGateN
		}
	}
	// Model-level: Embed(w+m+v) + ResidLambdas(w+m+v) + X0Lambdas(w+m+v)
	// + SmearGate(w+m+v) + SmearLambda(w+m+v) + BackoutLambda(w+m+v)
	embed := stories.Vocab * stories.Dim * 3
	residLambdas := stories.NLayers * 3
	x0Lambdas := stories.NLayers * 3
	smearGate := stories.Dim * stories.Dim * 3
	smearLambda := 1 * 3
	backoutLambda := 1 * 3
	return headerBytes + 4*(layerWeights+layerOpt+embed+residLambdas+x0Lambdas+smearGate+smearLambda+backoutLambda)
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
			layer.W1, layer.W2,
			layer.VEEmbed, layer.VEGate,
		} {
			if err := writeF32Slice(w, vals); err != nil {
				return err
			}
		}
	}
	if err := writeF32Slice(w, g.Embed); err != nil {
		return err
	}
	if err := writeF32Slice(w, g.ResidLambdas); err != nil {
		return err
	}
	if err := writeF32Slice(w, g.X0Lambdas); err != nil {
		return err
	}
	if err := writeF32Slice(w, g.SmearGate); err != nil {
		return err
	}
	if err := writeF32Slice(w, g.SmearLambda); err != nil {
		return err
	}
	return writeF32Slice(w, g.BackoutLambda)
}

func readModelGrad(r io.Reader, g *modelGrad) error {
	for i := range g.Layers {
		layer := &g.Layers[i]
		for _, vals := range [][]float32{
			layer.Wq, layer.Wk, layer.Wv, layer.Wo,
			layer.W1, layer.W2,
			layer.VEEmbed, layer.VEGate,
		} {
			if err := readF32Slice(r, vals); err != nil {
				return err
			}
		}
	}
	if err := readF32Slice(r, g.Embed); err != nil {
		return err
	}
	if err := readF32Slice(r, g.ResidLambdas); err != nil {
		return err
	}
	if err := readF32Slice(r, g.X0Lambdas); err != nil {
		return err
	}
	if err := readF32Slice(r, g.SmearGate); err != nil {
		return err
	}
	if err := readF32Slice(r, g.SmearLambda); err != nil {
		return err
	}
	return readF32Slice(r, g.BackoutLambda)
}

func drand48Seed(seed int64) uint64 {
	return ((uint64(seed) & 0xffffffff) << 16) | 0x330E
}

func (e *Engine) nextFloat64() float64 {
	e.rng = (drand48Mul*e.rng + drand48Add) & drand48Mask
	return float64(e.rng) / float64(uint64(1)<<48)
}

func (e *Engine) evalLogitsInto(tokens []uint16, logits []float32) error {
	if len(logits) != stories.Vocab*e.seq {
		return fmt.Errorf("storiesane eval logits: logits len=%d want=%d", len(logits), stories.Vocab*e.seq)
	}
	e.ensureOffload()
	if e.useANE && e.ensureLayers() == nil {
		err := e.evalLogitsANEInto(tokens, logits)
		if err == nil {
			return nil
		}
		e.disableLayerForward(err)
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

func (e *Engine) evalLogitsANEInto(tokens []uint16, logits []float32) error {
	e.ensureOffload()
	stories.EmbedLookup(e.x, e.mw.Embed, tokens, stories.Dim, e.seq)
	rmsNormNoWeightCF(e.x, e.x, stories.Dim, e.seq)
	smearForwardCF(e.x, e.mw.SmearGate, e.mw.SmearLambda[0], stories.Dim, e.seq)
	cur := e.x
	next := e.tmpHidden
	midLayer := stories.NLayers / 2
	for i := range e.layers {
		if err := e.layers[i].run(next, cur); err != nil {
			return fmt.Errorf("storiesane eval logits: layer %d: %w", i, err)
		}
		cur, next = next, cur
		if i == midLayer-1 {
			copy(e.xMid, cur)
		}
	}
	backoutForwardCF(cur, e.xMid, e.mw.BackoutLambda[0], stories.Dim, e.seq)
	stories.RMSNormNoWeight(e.xNorm, cur, stories.Dim, e.seq)
	if e.off == nil || !e.off.hasClassifierForward() {
		stories.MatMulVocabSeq(logits, e.mw.Embed, e.xNorm, stories.Vocab, stories.Dim, e.seq)
	} else if err := e.off.runClassifierForward(logits, e.xNorm); err != nil {
		return fmt.Errorf("storiesane eval logits: classifier: %w", err)
	}
	logitSoftcap(logits)
	return nil
}

func (e *Engine) evalLogitsCPUInto(tokens []uint16, logits []float32) error {
	x := e.x
	xNorm := e.xNorm
	next := e.tmpHidden
	qf := e.cpuQF
	kf := e.cpuKF
	vf := e.cpuVF
	attOut := e.cpuAttO
	x2 := e.cpuX2
	h1 := e.cpuH1
	gate := e.cpuGate
	ffOut := e.cpuFFOut
	veScr := e.cpuVEScr
	veGateAct := e.cpuVEGate
	x0 := e.cpuX0Inf

	stories.EmbedLookup(x, e.mw.Embed, tokens, stories.Dim, e.seq)
	rmsNormNoWeightCF(x, x, stories.Dim, e.seq)
	copy(x0, x)
	smearForwardCF(x, e.mw.SmearGate, e.mw.SmearLambda[0], stories.Dim, e.seq)
	cur := x
	midLayer := stories.NLayers / 2
	for i := range e.mw.Layers {
		layer := e.mw.Layers[i]
		rmsNormNoWeightCF(xNorm, cur, stories.Dim, e.seq)
		linearCF(qf, layer.Wq, xNorm, stories.Dim, stories.Dim, e.seq)
		linearCF(kf, layer.Wk, xNorm, stories.Dim, stories.Dim, e.seq)
		applyRoPECFInPlace(qf, stories.Heads, stories.Dim/stories.Heads, e.seq, e.ropeCos, e.ropeSin)
		applyRoPECFInPlace(kf, stories.Heads, stories.Dim/stories.Heads, e.seq, e.ropeCos, e.ropeSin)
		qkNormCF(qf, kf, stories.Dim, stories.Heads, e.seq)
		linearCF(vf, layer.Wv, xNorm, stories.Dim, stories.Dim, e.seq)
		if stories.IsVELayer(i) {
			veForwardCF(vf, veScr, veGateAct, layer.VEEmbed, layer.VEGate, cur, tokens, stories.Dim, e.seq)
		}
		causalAttentionCF(attOut, qf, kf, vf, stories.Heads, stories.Dim/stories.Heads, e.seq)
		linearCF(x2, layer.Wo, attOut, stories.Dim, stories.Dim, e.seq)
		rl := e.mw.ResidLambdas[i]
		xl := e.mw.X0Lambdas[i]
		n := stories.Dim * e.seq
		// x2 += rl * cur
		if !scaleAddAccel(x2, cur, rl, n) {
			for j := 0; j < n; j++ {
				x2[j] = rl*cur[j] + x2[j]
			}
		}
		if xl != 0 {
			// x2 += xl * x0
			if !scaleAddAccel(x2, x0, xl, n) {
				for j := 0; j < n; j++ {
					x2[j] += xl * x0[j]
				}
			}
		}
		rmsNormNoWeightCF(xNorm, x2, stories.Dim, e.seq)
		linearCF(h1, layer.W1, xNorm, stories.Hidden, stories.Dim, e.seq)
		for j := range gate {
			gate[j] = reluSquared32(h1[j])
		}
		linearCF(ffOut, layer.W2, gate, stories.Dim, stories.Hidden, e.seq)
		// next = ffOut + rl * x2
		copy(next[:n], ffOut[:n])
		if !scaleAddAccel(next, x2, rl, n) {
			for j := 0; j < n; j++ {
				next[j] = rl*x2[j] + ffOut[j]
			}
		}
		if xl != 0 {
			// next += xl * x0
			if !scaleAddAccel(next, x0, xl, n) {
				for j := 0; j < n; j++ {
					next[j] += xl * x0[j]
				}
			}
		}
		cur, next = next, cur
		if i == midLayer-1 {
			copy(e.xMid, cur)
		}
	}
	backoutForwardCF(cur, e.xMid, e.mw.BackoutLambda[0], stories.Dim, e.seq)
	stories.RMSNormNoWeight(e.xNorm, cur, stories.Dim, e.seq)
	stories.MatMulVocabSeq(logits, e.mw.Embed, e.xNorm, stories.Vocab, stories.Dim, e.seq)
	logitSoftcap(logits)
	return nil
}
