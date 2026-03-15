//go:build darwin

// Command generate runs autoregressive text generation on Apple Neural Engine.
//
// Usage:
//
//	go run ./cmd/generate --model stories110M.bin --prompt "Once upon a time"
//	go run ./cmd/generate --model qwen3-0.6b.bin --hf-tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/  --prompt "Once upon a time" --max-tokens 64
//	go run ./cmd/generate --model qwen3-4b.bin --prompt "1,2,3" --raw --max-tokens 16
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/tmc/autoresearch-go-ane/ane/stories"
	"github.com/tmc/autoresearch-go-ane/ane"
)

func main() {
	modelPath := flag.String("model", "stories110M.bin", "path to model .bin file")
	tokenizerPath := flag.String("tokenizer", "", "path to tokenizer.bin (TinyStories)")
	hfTokenizerDir := flag.String("hf-tokenizer", "", "path to HuggingFace model dir containing tokenizer.json")
	hfModelID := flag.String("hf-model-id", "", "HuggingFace model ID for Python-backed encoding (e.g. Qwen/Qwen3-0.6B)")
	prompt := flag.String("prompt", "Once upon a time", "input prompt text or comma-separated token IDs")
	maxTokens := flag.Int("max-tokens", 64, "maximum tokens to generate")
	temperature := flag.Float64("temperature", 0.8, "sampling temperature (0 = greedy)")
	topP := flag.Float64("top-p", 0.9, "nucleus sampling threshold (0 = disabled)")
	seqLen := flag.Int("seq", 256, "sequence length for the model")
	useANE := flag.Bool("ane", true, "use Apple Neural Engine")
	rawTokens := flag.Bool("raw", false, "treat --prompt as comma-separated token IDs")
	cpuProfile := flag.String("cpuprofile", "", "write CPU profile to file")
	memProfile := flag.String("memprofile", "", "write memory profile to file")
	flag.Parse()

	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	if *modelPath == "" {
		log.Fatal("--model is required")
	}

	// Auto-detect HF tokenizer from model path if not specified.
	if *hfTokenizerDir == "" && *tokenizerPath == "" && !*rawTokens {
		// Try to find tokenizer.json near the model or in HF cache.
		if dir := findHFTokenizerDir(*modelPath); dir != "" {
			*hfTokenizerDir = dir
		}
	}

	log.Printf("loading model from %s...", *modelPath)
	t0 := time.Now()

	opts := ane.Options{
		ModelPath:         *modelPath,
		Seq:               *seqLen,
		UseANE:            *useANE,
		CPUClassifierHead: true,
	}
	engine, err := ane.Open(opts)
	if err != nil {
		log.Fatalf("open engine: %v", err)
	}
	defer engine.Close()

	cfg := engine.Config()
	log.Printf("model loaded in %v: dim=%d, hidden=%d, heads=%d, kv_heads=%d, layers=%d, vocab=%d",
		time.Since(t0).Round(time.Millisecond),
		cfg.Dim, cfg.Hidden, cfg.Heads, cfg.EffectiveKVHeads(), cfg.NLayers, cfg.Vocab)

	// Load HF tokenizer if available.
	var hfTok *stories.HFTokenizer
	if *hfTokenizerDir != "" {
		// Expand glob in path (for ~/.cache/huggingface/hub/models--*/snapshots/*/)
		matches, _ := filepath.Glob(*hfTokenizerDir)
		dir := *hfTokenizerDir
		if len(matches) > 0 {
			dir = matches[0]
		}
		var tokErr error
		hfTok, tokErr = stories.LoadHFTokenizer(dir)
		if tokErr != nil {
			log.Printf("warning: failed to load HF tokenizer from %s: %v", dir, tokErr)
		} else {
			log.Printf("loaded HF tokenizer: vocab=%d", hfTok.VocabSize())
			if *hfModelID != "" {
				hfTok.SetModelID(*hfModelID)
			}
		}
	}

	// Build prompt tokens.
	var promptTokens []int32
	if *rawTokens {
		promptTokens = parseRawTokens(*prompt)
	} else if hfTok != nil {
		// Go-native BPE encoding — no Python needed.
		promptTokens, err = hfTok.Encode(*prompt)
		if err != nil {
			log.Fatalf("encode prompt: %v", err)
		}
	} else if *tokenizerPath != "" {
		tok, err := stories.LoadTokenizer(*tokenizerPath)
		if err != nil {
			log.Fatalf("load tokenizer: %v", err)
		}
		_ = tok
		promptTokens = byteEncode(*prompt)
	} else {
		promptTokens = byteEncode(*prompt)
	}

	if len(promptTokens) == 0 {
		promptTokens = []int32{1} // BOS
	}

	log.Printf("prompt: %d tokens", len(promptTokens))

	// Print the prompt first.
	fmt.Print(*prompt)

	// Set EOS token based on model vocab size.
	eosToken := int32(2) // default for TinyStories
	if cfg.Vocab > 32000 {
		// Qwen3 EOS token ID is 151645.
		eosToken = 151645
	}

	genOpts := stories.GenerateOptions{
		MaxTokens:   *maxTokens,
		Temperature: float32(*temperature),
		TopP:        float32(*topP),
		StopTokens:  []int32{eosToken},
	}

	generated := 0
	genStart := time.Now()

	err = ane.GenerateStream(engine, promptTokens, genOpts, func(token int32, step int) bool {
		generated++

		if hfTok != nil {
			// Go-native decoding — fast, no Python subprocess.
			fmt.Print(hfTok.DecodeToken(int(token)))
		} else if *tokenizerPath != "" {
			tok, loadErr := stories.LoadTokenizer(*tokenizerPath)
			if loadErr == nil {
				fmt.Print(tok.Decode(int(token)))
			} else {
				fmt.Printf("[%d]", token)
			}
		} else {
			if token < 128 && token >= 32 {
				fmt.Print(string(rune(token)))
			} else {
				fmt.Printf("[%d]", token)
			}
		}

		if generated <= 3 {
			t := engine.LastTokenTimings()
			log.Printf("  token %d timings: total=%v qkv=%v attn=%v wo=%v ffn=%v cls=%v fp16=%v",
				step, t.Total.Round(time.Millisecond), t.QKV.Round(time.Millisecond),
				t.Attention.Round(time.Millisecond), t.Wo.Round(time.Millisecond),
				t.FFN.Round(time.Millisecond), t.Classifier.Round(time.Millisecond),
				t.FP16Conv.Round(time.Millisecond))
		}

		return true
	})
	if err != nil {
		log.Fatalf("\ngeneration error: %v", err)
	}

	elapsed := time.Since(genStart)
	tokPerSec := float64(generated) / elapsed.Seconds()

	if *memProfile != "" {
		f, err := os.Create(*memProfile)
		if err != nil {
			log.Fatal(err)
		}
		runtime.GC()
		pprof.WriteHeapProfile(f)
		f.Close()
	}

	fmt.Println()
	log.Printf("generated %d tokens in %v (%.1f tok/s)", generated, elapsed.Round(time.Millisecond), tokPerSec)
}

// parseRawTokens parses comma-separated token IDs.
func parseRawTokens(s string) []int32 {
	parts := strings.Split(s, ",")
	tokens := make([]int32, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		var v int
		if _, err := fmt.Sscanf(p, "%d", &v); err == nil && v >= 0 {
			tokens = append(tokens, int32(v))
		}
	}
	return tokens
}

// byteEncode converts a string to token IDs using byte values.
func byteEncode(s string) []int32 {
	tokens := make([]int32, 0, len(s)+1)
	tokens = append(tokens, 1) // BOS
	for _, b := range []byte(s) {
		tokens = append(tokens, int32(b))
	}
	return tokens
}

// findHFTokenizerDir tries to locate a HuggingFace tokenizer directory
// based on the model file name.
func findHFTokenizerDir(modelPath string) string {
	base := filepath.Base(modelPath)
	base = strings.TrimSuffix(base, ".bin")

	// Map common model names to HF model IDs.
	nameMap := map[string]string{
		"qwen3-0.6b": "Qwen--Qwen3-0.6B",
		"qwen3-4b":   "Qwen--Qwen3-4B",
	}

	hfName, ok := nameMap[strings.ToLower(base)]
	if !ok {
		return ""
	}

	pattern := filepath.Join(
		filepath.Dir(modelPath),
		"..", ".cache", "huggingface", "hub",
		"models--"+hfName, "snapshots", "*",
	)
	matches, _ := filepath.Glob(pattern)
	if len(matches) > 0 {
		return matches[0]
	}

	// Try home directory cache.
	home, _ := filepath.Glob(filepath.Join("~", ".cache", "huggingface", "hub", "models--"+hfName, "snapshots", "*"))
	if len(home) > 0 {
		return home[0]
	}

	return ""
}
