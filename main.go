//go:build darwin

package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/tmc/anperf"
	"github.com/tmc/autoresearch-go-ane/ane/stories"
	"github.com/tmc/autoresearch-go-ane/ane/storiesane"
)

func main() {
	modelPath := flag.String("model", "stories110M.bin", "path to model checkpoint (or random-init target)")
	dataPath := flag.String("data", "tinystories_data00.bin", "path to token data file")
	evalOnly := flag.Bool("eval-only", false, "skip training, just evaluate")
	timeBudget := flag.Duration("time-budget", 5*time.Minute, "training wall-clock budget")
	resultsPath := flag.String("results", "results.tsv", "path to results TSV file")
	randomInit := flag.Bool("random-init", true, "start from random weights (ignore pretrained)")
	flag.Parse()

	log.Println("loading tokens from", *dataPath)
	tokens, err := loadTokens(*dataPath)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("loaded %d tokens (%.1f MB)", len(tokens), float64(len(tokens)*2)/(1024*1024))

	if *randomInit {
		initPath := strings.TrimSuffix(*modelPath, ".bin") + "_randinit.bin"
		log.Println("ensuring random-init checkpoint at", initPath)
		if err := ensureRandomInitModel(initPath, Seed); err != nil {
			log.Fatal(err)
		}
		*modelPath = initPath
	}

	opts := experimentConfig(*modelPath, tokens)
	log.Println("opening engine...")
	engine, err := storiesane.Open(opts)
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()

	// Start anperf dashboard.
	perf := anperf.New(
		anperf.WithCapacity(100000),
		anperf.WithDataDir(".anperf"),
		anperf.WithRunName(gitCommit()),
		anperf.WithSystemInfo(anperf.SystemInfo{
			Chip:     "Apple Silicon",
			ANECores: 16,
		}),
	)
	perf.SetHyperparams(map[string]interface{}{
		"seq_length":     SequenceLength,
		"accum_steps":    AccumSteps,
		"learning_rate":  LearningRate,
		"adam_beta1":     AdamBeta1,
		"adam_beta2":     AdamBeta2,
		"weight_decay":   WeightDecay,
		"grad_clip":      GradClip,
		"loss_scale":     LossScale,
		"use_ane":        UseANE,
		"hybrid_backward": HybridBackward,
		"seed":           Seed,
	})
	http.Handle("/perf/", perf.Handler("/perf/"))
	go func() {
		log.Println("anperf dashboard: http://localhost:9090/perf/")
		if err := http.ListenAndServe(":9090", nil); err != nil {
			log.Printf("anperf server: %v", err)
		}
	}()

	if *evalOnly {
		loss, err := evalLoss(engine, tokens)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("val_loss:         %.6f\n", loss)
		return
	}

	// Training loop.
	log.Printf("training for %v (excluding warmup steps)", *timeBudget)

	var (
		steps      int
		warmupDone bool
		trainStart time.Time
	)
	const warmupSteps = 3

	for {
		res, err := engine.Step()
		if err != nil {
			log.Printf("step %d error: %v", steps, err)
			logResult(*resultsPath, 0, steps, 0, "crash", fmt.Sprintf("step error: %v", err))
			os.Exit(1)
		}
		steps++

		// Record real metrics to anperf.
		msOf := func(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }
		stepMs := msOf(res.StepDuration)
		tokPerSec := float64(SequenceLength) / (stepMs / 1000.0)
		perf.Record(anperf.StepResult{
			Step:         steps,
			Loss:         res.Loss,
			StepMs:       stepMs,
			ANEMs:        msOf(res.ANEEvalDuration),
			AdamMs:       msOf(res.AdamDuration),
			CPUMs:        msOf(res.CPUWorkDuration),
			CompileMs:    msOf(res.CompileDuration),
			TokensPerSec: tokPerSec,
			Components: map[string]float64{
				"Compile":       msOf(res.CompileDuration),
				"StartupCompile": msOf(res.StartupCompileDuration),
				"ANEEval":       msOf(res.ANEEvalDuration),
				"FinalHead":     msOf(res.FinalHeadDuration),
				"EmbedGrad":     msOf(res.EmbedGradDuration),
				"RMSDW":         msOf(res.RMSDWDuration),
				"DWGEMM":        msOf(res.DWGEMMDuration),
				"DWWait":        msOf(res.DWWaitDuration),
				"WeightRefresh": msOf(res.WeightRefreshDuration),
				"Adam":          msOf(res.AdamDuration),
			},
		})

		if !warmupDone && steps > warmupSteps {
			warmupDone = true
			trainStart = time.Now()
			log.Println("warmup complete, starting timed training")
		}

		if warmupDone {
			elapsed := time.Since(trainStart)
			progress := min(elapsed.Seconds()/timeBudget.Seconds(), 1.0)
			lr := lrSchedule(progress)
			if err := engine.SetLR(float32(lr)); err != nil {
				log.Printf("set LR: %v", err)
			}
			if steps%10 == 0 {
				log.Printf("step %d | loss %.4f | lr %.2e | elapsed %v", steps, res.Loss, lr, elapsed.Round(time.Second))
			}
			if elapsed >= *timeBudget {
				log.Printf("time budget reached after %d steps (%.1fs)", steps, elapsed.Seconds())
				break
			}
		} else {
			log.Printf("warmup step %d/%d | loss %.4f | compile %v", steps, warmupSteps, res.Loss, res.CompileDuration)
		}
	}

	trainElapsed := time.Since(trainStart)

	log.Println("evaluating...")
	valLoss, err := evalLoss(engine, tokens)
	if err != nil {
		log.Printf("eval error: %v", err)
		logResult(*resultsPath, 0, steps, trainElapsed.Seconds(), "eval_error", err.Error())
		os.Exit(1)
	}

	fmt.Printf("val_loss:         %.6f\n", valLoss)
	fmt.Printf("training_seconds: %.1f\n", trainElapsed.Seconds())
	fmt.Printf("steps:            %d\n", steps)
	fmt.Printf("sequence_length:  %d\n", SequenceLength)
	fmt.Printf("accum_steps:      %d\n", AccumSteps)
	fmt.Printf("learning_rate:    %.1e\n", LearningRate)
	fmt.Printf("use_ane:          %v\n", UseANE)
	fmt.Printf("hybrid_backward:  %v\n", HybridBackward)
	fmt.Printf("seed:             %d\n", Seed)

	logResult(*resultsPath, valLoss, steps, trainElapsed.Seconds(), "ok", "baseline")

	// Auto-save anperf snapshot.
	if snap, err := perf.SaveSnapshot(gitCommit(), gitCommit(), nil); err == nil {
		log.Printf("anperf snapshot saved: %s (ANE util %.1f%%, %.0f tok/s)", snap.ID, snap.ANEUtil, snap.AvgTokensSec)
	}

	// Keep anperf dashboard alive after training completes.
	log.Println("training done — dashboard still serving at http://localhost:9090/perf/")
	select {}
}

// --- token loading ---

func loadTokens(path string) ([]uint16, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokens: %w", err)
	}
	if len(data)%2 != 0 {
		return nil, fmt.Errorf("read tokens: file size %d is not even", len(data))
	}
	n := len(data) / 2
	tokens := make([]uint16, n)
	for i := range tokens {
		tokens[i] = binary.LittleEndian.Uint16(data[i*2:])
	}
	return tokens, nil
}

// --- evaluation ---

const (
	evalTokens = 100_000
	evalSeqLen = 256
)

func evalLoss(engine *storiesane.Engine, tokens []uint16) (float64, error) {
	if len(tokens) < evalTokens+evalSeqLen {
		return 0, fmt.Errorf("eval: need at least %d tokens, got %d", evalTokens+evalSeqLen, len(tokens))
	}

	valTokens := tokens[len(tokens)-evalTokens:]

	var totalLoss float64
	var totalPositions int

	for pos := 0; pos+evalSeqLen < len(valTokens); pos += evalSeqLen - 1 {
		window := valTokens[pos : pos+evalSeqLen]
		logits, err := engine.EvalLogits(window)
		if err != nil {
			return 0, fmt.Errorf("eval logits at pos %d: %w", pos, err)
		}

		vocabSize := len(logits) / evalSeqLen
		for i := 0; i < evalSeqLen-1; i++ {
			row := logits[i*vocabSize : (i+1)*vocabSize]

			maxVal := float64(math.SmallestNonzeroFloat64)
			for _, v := range row {
				if float64(v) > maxVal {
					maxVal = float64(v)
				}
			}
			var sumExp float64
			for _, v := range row {
				sumExp += math.Exp(float64(v) - maxVal)
			}

			target := window[i+1]
			logProb := float64(row[target]) - maxVal - math.Log(sumExp)
			totalLoss -= logProb
			totalPositions++
		}
	}

	if totalPositions == 0 {
		return 0, fmt.Errorf("eval: no positions evaluated")
	}
	return totalLoss / float64(totalPositions), nil
}

// --- random init ---

func ensureRandomInitModel(path string, seed int64) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	mw := stories.NewModelWeights(stories.Vocab)
	stories.RandomInit(mw, seed)
	opt := stories.NewOptimState(stories.Vocab)
	meta := stories.TrainMeta{Step: 0, LR: LearningRate}
	if err := stories.SaveCheckpoint(path, meta, mw, opt); err != nil {
		return fmt.Errorf("save random init checkpoint: %w", err)
	}
	return nil
}

// --- results logging ---

type result struct {
	Commit, Status, Description string
	ValLoss, TrainSecs          float64
	Steps                       int
}

func logResult(path string, valLoss float64, steps int, secs float64, status, desc string) {
	commit := gitCommit()
	if err := appendResult(path, result{commit, status, desc, valLoss, secs, steps}); err != nil {
		log.Printf("failed to log result: %v", err)
	}
}

func appendResult(path string, r result) error {
	info, err := os.Stat(path)
	needHeader := os.IsNotExist(err) || (err == nil && info.Size() == 0)

	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("open results: %w", err)
	}
	defer f.Close()

	if needHeader {
		fmt.Fprintf(f, "commit\tval_loss\tsteps\ttrain_secs\tstatus\tdescription\n")
	}
	fmt.Fprintf(f, "%s\t%.6f\t%d\t%.1f\t%s\t%s\n",
		r.Commit, r.ValLoss, r.Steps, r.TrainSecs, r.Status, r.Description)
	return nil
}

func gitCommit() string {
	out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}
