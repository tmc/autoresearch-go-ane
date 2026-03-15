//go:build darwin

// experiment.go — THE MODIFIABLE FILE
//
// The autonomous agent edits this file to explore inference configuration,
// dispatch strategies, and acceleration options. All other files in this
// package are read-only.

package main

import (
	"math"

	"github.com/tmc/autoresearch-go-ane/ane"
)

// Inference configuration — the primary tuning surface.
const (
	SequenceLength = 256 // context window for benchmarking

	// Acceleration dispatch flags.
	UseMetal    = false // Metal GPU matmul for large layers
	UseBNNS     = false // BNNS fp16-weight GEMV
	UseANE      = false // Apple Neural Engine acceleration (disabled: ANE framework incompatible)
	TileSize    = 0     // tiled layer dimension (0 = auto)

	// Buffer management.
	PreallocBuffers = true // pre-allocate scratch buffers at engine init
)

// Training constants — kept for backward compatibility with harness.
const (
	AccumSteps     = 4
	LearningRate   = 3e-4
	AdamBeta1      = 0.9
	AdamBeta2      = 0.999
	AdamEps        = 1e-8
	WeightDecay    = 0.01
	GradClip       = 1.0
	LossScale      = 256.0
	HybridBackward = true
	Seed           = 42
)

const warmupFraction = 0.05

func experimentConfig(modelPath string, tokens []int32) ane.Options {
	return ane.Options{
		ModelPath:      modelPath,
		Tokens:         tokens,
		Seq:            SequenceLength,
		AccumSteps:     AccumSteps,
		LR:             LearningRate,
		Seed:           Seed,
		AdamBeta1:      AdamBeta1,
		AdamBeta2:      AdamBeta2,
		AdamEps:        AdamEps,
		WeightDecay:    WeightDecay,
		GradClip:       GradClip,
		LossScale:      LossScale,
		UseANE:            UseANE,
		CPUClassifierHead: false,
		HybridBackward:    HybridBackward,
	}
}

// lrSchedule returns the learning rate for the given training progress (0..1).
// Linear warmup followed by cosine decay to 10% of peak LR.
func lrSchedule(progress float64) float64 {
	if progress < warmupFraction {
		return LearningRate * progress / warmupFraction
	}
	decay := 0.5 * (1.0 + math.Cos(math.Pi*(progress-warmupFraction)/(1.0-warmupFraction)))
	return LearningRate * (0.1 + 0.9*decay)
}
