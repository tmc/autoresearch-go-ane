//go:build darwin

// Command generate runs autoregressive text generation on Apple Neural Engine.
//
// Usage:
//
//	go run ./cmd/generate --model stories110M.bin --prompt "Once upon a time"
//	go run ./cmd/generate --model qwen3-4b.bin --prompt "def fibonacci(n):" --max-tokens 128
package main

import (
	"flag"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/tmc/autoresearch-go-ane/ane/stories"
	"github.com/tmc/autoresearch-go-ane/ane/storiesane"
)

func main() {
	modelPath := flag.String("model", "stories110M.bin", "path to model .bin file")
	tokenizerPath := flag.String("tokenizer", "", "path to tokenizer.bin (optional, for TinyStories)")
	prompt := flag.String("prompt", "Once upon a time", "input prompt text or comma-separated token IDs")
	maxTokens := flag.Int("max-tokens", 64, "maximum tokens to generate")
	temperature := flag.Float64("temperature", 0.8, "sampling temperature (0 = greedy)")
	topP := flag.Float64("top-p", 0.9, "nucleus sampling threshold (0 = disabled)")
	seqLen := flag.Int("seq", 256, "sequence length for the model")
	useANE := flag.Bool("ane", true, "use Apple Neural Engine")
	rawTokens := flag.Bool("raw", false, "treat --prompt as comma-separated uint16 token IDs")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("--model is required")
	}

	log.Printf("loading model from %s...", *modelPath)
	t0 := time.Now()

	opts := storiesane.Options{
		ModelPath:         *modelPath,
		Seq:               *seqLen,
		UseANE:            *useANE,
		CPUClassifierHead: true,
	}
	engine, err := storiesane.Open(opts)
	if err != nil {
		log.Fatalf("open engine: %v", err)
	}
	defer engine.Close()

	cfg := engine.Config()
	log.Printf("model loaded in %v: dim=%d, hidden=%d, heads=%d, kv_heads=%d, layers=%d, vocab=%d",
		time.Since(t0).Round(time.Millisecond),
		cfg.Dim, cfg.Hidden, cfg.Heads, cfg.EffectiveKVHeads(), cfg.NLayers, cfg.Vocab)

	// Build prompt tokens.
	var promptTokens []uint16
	if *rawTokens {
		promptTokens = parseRawTokens(*prompt)
	} else if *tokenizerPath != "" {
		tok, err := stories.LoadTokenizer(*tokenizerPath)
		if err != nil {
			log.Fatalf("load tokenizer: %v", err)
		}
		// Simple character-level encoding for TinyStories tokenizer.
		promptTokens = encodeWithTokenizer(tok, *prompt)
	} else {
		// Fall back to byte-level encoding: each byte becomes a token.
		// This works reasonably for models with byte-level vocabularies.
		promptTokens = byteEncode(*prompt)
	}

	if len(promptTokens) == 0 {
		promptTokens = []uint16{1} // BOS
	}

	log.Printf("prompt: %d tokens", len(promptTokens))

	// Print prompt tokens as context.
	if *tokenizerPath != "" {
		fmt.Print(*prompt)
	}

	genOpts := stories.GenerateOptions{
		MaxTokens:   *maxTokens,
		Temperature: float32(*temperature),
		TopP:        float32(*topP),
		StopTokens:  []uint16{2}, // EOS
	}

	generated := 0
	genStart := time.Now()

	err = storiesane.GenerateStream(engine, promptTokens, genOpts, func(token uint16, step int) bool {
		generated++

		if *tokenizerPath != "" {
			// Use tokenizer to decode.
			tok, loadErr := stories.LoadTokenizer(*tokenizerPath)
			if loadErr == nil {
				fmt.Print(tok.Decode(int(token)))
			} else {
				fmt.Printf("[%d]", token)
			}
		} else {
			// Print raw token ID.
			if token < 128 && token >= 32 {
				fmt.Print(string(rune(token)))
			} else {
				fmt.Printf("[%d]", token)
			}
		}

		return true
	})
	if err != nil {
		log.Fatalf("\ngeneration error: %v", err)
	}

	elapsed := time.Since(genStart)
	tokPerSec := float64(generated) / elapsed.Seconds()
	fmt.Println()
	log.Printf("generated %d tokens in %v (%.1f tok/s)", generated, elapsed.Round(time.Millisecond), tokPerSec)
}

// parseRawTokens parses comma-separated token IDs.
func parseRawTokens(s string) []uint16 {
	parts := strings.Split(s, ",")
	tokens := make([]uint16, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		var v int
		if _, err := fmt.Sscanf(p, "%d", &v); err == nil && v >= 0 && v < 65536 {
			tokens = append(tokens, uint16(v))
		}
	}
	return tokens
}

// byteEncode converts a string to token IDs using byte values.
// This is a minimal fallback for models without a loaded tokenizer.
func byteEncode(s string) []uint16 {
	tokens := make([]uint16, 0, len(s)+1)
	tokens = append(tokens, 1) // BOS
	for _, b := range []byte(s) {
		tokens = append(tokens, uint16(b))
	}
	return tokens
}

// encodeWithTokenizer does a simple greedy longest-match encode.
// For proper encoding of Qwen3 etc., use the Python tokenizer wrapper.
func encodeWithTokenizer(tok *stories.Tokenizer, text string) []uint16 {
	// Simple: just use byte encoding as fallback for now.
	// A real tokenizer would need BPE/SentencePiece logic.
	_ = tok
	return byteEncode(text)
}

// DecodeOptions is unused but reserved for future tokenizer config.
type DecodeOptions struct{}
