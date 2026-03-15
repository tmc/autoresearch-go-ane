package stories

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// HFTokenizer wraps a HuggingFace tokenizer loaded from tokenizer.json.
// It provides fast Go-native decoding and shells out to Python for encoding.
type HFTokenizer struct {
	vocab    map[int]string // id → token string
	invVocab map[string]int // token string → id
	vocabSize int
	modelID   string // HF model ID for Python fallback encoding
}

// LoadHFTokenizer loads a HuggingFace tokenizer.json file.
// The modelDir should contain tokenizer.json and optionally tokenizer_config.json.
func LoadHFTokenizer(modelDir string) (*HFTokenizer, error) {
	tokPath := filepath.Join(modelDir, "tokenizer.json")
	data, err := os.ReadFile(tokPath)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer.json: %w", err)
	}

	// Parse the tokenizer.json format used by HuggingFace tokenizers.
	var tokJSON struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &tokJSON); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}

	vocab := make(map[int]string, len(tokJSON.Model.Vocab))
	invVocab := make(map[string]int, len(tokJSON.Model.Vocab))
	maxID := 0
	for token, id := range tokJSON.Model.Vocab {
		vocab[id] = token
		invVocab[token] = id
		if id > maxID {
			maxID = id
		}
	}
	// Add special/added tokens.
	for _, at := range tokJSON.AddedTokens {
		vocab[at.ID] = at.Content
		invVocab[at.Content] = at.ID
		if at.ID > maxID {
			maxID = at.ID
		}
	}

	return &HFTokenizer{
		vocab:     vocab,
		invVocab:  invVocab,
		vocabSize: maxID + 1,
	}, nil
}

// SetModelID sets the HuggingFace model ID used for Python-backed encoding.
func (t *HFTokenizer) SetModelID(modelID string) {
	t.modelID = modelID
}

// VocabSize returns the vocabulary size.
func (t *HFTokenizer) VocabSize() int {
	return t.vocabSize
}

// DecodeToken converts a single token ID to its string representation.
func (t *HFTokenizer) DecodeToken(id int) string {
	if t == nil {
		return ""
	}
	s, ok := t.vocab[id]
	if !ok {
		return ""
	}
	// Handle byte-level BPE encoding: replace Unicode byte chars with actual bytes.
	return decodeBPEToken(s)
}

// DecodeTokens converts multiple token IDs to a single string.
func (t *HFTokenizer) DecodeTokens(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		b.WriteString(t.DecodeToken(id))
	}
	return b.String()
}

// Encode tokenizes text into token IDs using the Python tokenizer.
// This shells out to Python for encoding, which is slower but correct.
// For batch encoding (prompt at startup), this is acceptable.
func (t *HFTokenizer) Encode(text string) ([]uint16, error) {
	if t.modelID == "" {
		return nil, fmt.Errorf("hf tokenizer: model ID not set, call SetModelID first")
	}

	cmd := exec.Command("python3", "-c", fmt.Sprintf(
		`from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained(%q); ids = t.encode(%q); print(",".join(str(i) for i in ids))`,
		t.modelID, text))
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("hf tokenizer encode: %w", err)
	}

	parts := strings.Split(strings.TrimSpace(string(out)), ",")
	tokens := make([]uint16, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("hf tokenizer encode: parse token %q: %w", p, err)
		}
		tokens = append(tokens, uint16(v))
	}
	return tokens, nil
}

// decodeBPEToken handles the byte-level BPE encoding used by GPT-2/Qwen tokenizers.
// These tokenizers map byte values to Unicode characters to avoid control characters.
func decodeBPEToken(s string) string {
	// The byte-to-unicode mapping used by GPT-2 style tokenizers.
	// Check if any character is in the mapped range and decode accordingly.
	runes := []rune(s)
	bytes := make([]byte, 0, len(runes))
	for _, r := range runes {
		b, ok := unicodeToByte(r)
		if ok {
			bytes = append(bytes, b)
		} else {
			// Non-BPE character, just append UTF-8.
			bytes = append(bytes, []byte(string(r))...)
		}
	}
	return string(bytes)
}

// unicodeToByte reverses the GPT-2 byte-to-unicode mapping.
func unicodeToByte(r rune) (byte, bool) {
	// Direct ASCII printable range: ! (33) through ~ (126)
	if r >= '!' && r <= '~' {
		return byte(r), true
	}
	// ¡ (161) through ¬ (172)
	if r >= '¡' && r <= '¬' {
		return byte(r), true
	}
	// ® (174) through ÿ (255)
	if r >= '®' && r <= 'ÿ' {
		return byte(r), true
	}
	// Mapped characters for bytes 0-32, 127-160, 173:
	// These use Unicode characters starting at Ā (256).
	if r >= 'Ā' && r <= 'Ġ' {
		// Maps to bytes 0-32 (Ā=0, ā=1, ..., Ġ=32)
		return byte(r - 'Ā'), true
	}
	if r == 'ġ' {
		return 127, true
	}
	if r >= 'Ģ' && r <= 'ģ'+32 {
		// Maps to bytes 128-160
		return byte(128 + (r - 'Ģ')), true
	}
	// 173 maps to a specific char
	if r == 'Ń' {
		return 173, true
	}
	return 0, false
}
