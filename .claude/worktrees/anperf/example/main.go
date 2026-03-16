// example/main.go — demonstrates anperf integration with simulated training data.
//
// Run with: go run ./example
// Then open http://localhost:9090/perf/ in your browser.
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"time"

	"github.com/anthropics/anperf"
)

func main() {
	// Create a recorder with system info.
	perf := anperf.New(
		anperf.WithCapacity(50000),
		anperf.WithSystemInfo(anperf.SystemInfo{
			Chip:     "Apple M4 Max",
			Memory:   "128 GB Unified",
			ANECores: 16,
		}),
	)

	perf.SetHyperparams(map[string]interface{}{
		"lr":          3e-4,
		"seq":         256,
		"accum_steps": 4,
		"adam_beta1":  0.9,
		"adam_beta2":  0.999,
		"weight_decay": 0.01,
		"grad_clip":  1.0,
		"use_ane":    true,
	})

	// Serve the dashboard.
	http.Handle("/perf/", perf.Handler("/perf/"))

	// Redirect root to dashboard.
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			http.Redirect(w, r, "/perf/", http.StatusTemporaryRedirect)
			return
		}
		http.NotFound(w, r)
	})

	log.Println("anperf dashboard: http://localhost:9090/perf/")

	// Start server in background.
	go func() {
		if err := http.ListenAndServe(":9090", nil); err != nil {
			log.Fatal(err)
		}
	}()

	// Simulate training loop.
	rng := rand.New(rand.NewSource(42))
	baseLoss := float32(10.5)
	baseStepMs := 55.0
	baseANEMs := 32.0
	baseAdamMs := 12.0
	baseCPUMs := 8.0
	seqLen := 256

	for step := 1; step <= 20000; step++ {
		// Simulate loss decay with noise.
		progress := float64(step) / 20000.0
		loss := baseLoss * float32(math.Exp(-3.0*progress)) * float32(1.0+0.05*rng.NormFloat64())
		if loss < 0.1 {
			loss = 0.1
		}

		// Simulate timing variations.
		stepMs := baseStepMs * (1.0 + 0.1*rng.NormFloat64())
		aneMs := baseANEMs * (1.0 + 0.08*rng.NormFloat64())
		adamMs := baseAdamMs * (1.0 + 0.15*rng.NormFloat64())
		cpuMs := baseCPUMs * (1.0 + 0.12*rng.NormFloat64())
		compileMs := 0.0
		if step <= 3 {
			compileMs = 200 + 100*rng.Float64()
		}

		tokensPerSec := float64(seqLen) / (stepMs / 1000.0)

		// Component timings.
		components := map[string]float64{
			"Compile":      compileMs,
			"FinalHead":    6.0 + 2.0*rng.NormFloat64(),
			"EmbedGrad":    3.5 + 1.0*rng.NormFloat64(),
			"RMSDW":        1.2 + 0.3*rng.NormFloat64(),
			"DWGEMM":       8.0 + 2.0*rng.NormFloat64(),
			"DWWait":       2.0 + 0.5*rng.NormFloat64(),
			"WeightRefresh": 1.5 + 0.5*rng.NormFloat64(),
		}

		// LR schedule: warmup + cosine decay.
		lr := 3e-4
		warmupFrac := 0.05
		if progress < warmupFrac {
			lr = 3e-4 * progress / warmupFrac
		} else {
			decay := 0.5 * (1.0 + math.Cos(math.Pi*(progress-warmupFrac)/(1.0-warmupFrac)))
			lr = 3e-4 * (0.1 + 0.9*decay)
		}

		valLoss := float32(0)
		if step%100 == 0 {
			valLoss = loss * float32(1.0+0.1*rng.Float64())
		}

		perf.Record(anperf.StepResult{
			Step:         step,
			Loss:         loss,
			ValLoss:      valLoss,
			StepMs:       stepMs,
			ANEMs:        aneMs,
			AdamMs:       adamMs,
			CPUMs:        cpuMs,
			CompileMs:    compileMs,
			TokensPerSec: tokensPerSec,
			LR:           lr,
			Components:   components,
		})

		// Record experiments periodically.
		if step%2000 == 0 {
			perf.RecordExperiment(
				fmt.Sprintf("run_%d", step/2000),
				map[string]interface{}{
					"lr":         lr,
					"steps":      step,
					"accum":      4,
					"grad_clip":  1.0,
				},
				float64(loss),
				time.Duration(step)*55*time.Millisecond,
				step,
				"ok",
			)
		}

		// Simulate real training pace (~20 steps/sec for demo).
		time.Sleep(50 * time.Millisecond)
	}

	log.Println("training complete")
	select {} // Keep server running.
}
