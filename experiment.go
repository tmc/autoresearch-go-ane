//go:build darwin

// experiment.go — THE MODIFIABLE FILE
//
// The autonomous agent edits this file to explore hyperparameters,
// learning rate schedules, and training configurations. All other
// files in this package are read-only.

package main

import (
	"math"

	"github.com/tmc/autoresearch-go-ane/ane"
)

const (
	SequenceLength = 256
	AccumSteps     = 8
	LearningRate   = 3e-4
	AdamBeta1      = 0.8
	AdamBeta2      = 0.95
	AdamEps        = 1e-10
	WeightDecay    = 0.0
	GradClip       = 1.0
	LossScale      = 256.0
	UseANE         = true
	HybridBackward = true
	Seed           = 42

	// Per-param-group LR multipliers (relative to LearningRate).
	EmbedLRMult  = float32(1.0)  // embed params (Embed, VEEmbed)
	ScalarLRMult = float32(1.0)  // scalar params (VEGate, SmearLambda, BackoutLambda)
	LambdaLRMult = float32(0.01) // lambda params (ResidLambdas, X0Lambdas), relative to scalar LR

	// Custom betas for lambda params (ResidLambdas, X0Lambdas).
	LambdaBeta1 = float32(0.96)
	LambdaBeta2 = float32(0.95)
)

const warmupFraction = 0.05

func experimentConfig(modelPath string, tokens []uint16) ane.Options {
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
		UseANE:         UseANE,
		HybridBackward: HybridBackward,
		EmbedLRMult:    EmbedLRMult,
		ScalarLRMult:   ScalarLRMult,
		LambdaLRMult:   LambdaLRMult,
		LambdaBeta1:    LambdaBeta1,
		LambdaBeta2:    LambdaBeta2,
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
