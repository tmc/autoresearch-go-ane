package stories

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// HFTokenizer implements byte-level BPE tokenization for HuggingFace models.
// Adapted from github.com/tmc/mlx-go's tokenize package.
//
// Supports GPT-2 style byte-level BPE (Qwen3, Llama, GPT) and SentencePiece
// style BPE (Gemma, Mistral). Loads from tokenizer.json format.
type HFTokenizer struct {
	vocab        map[string]int // token string → ID
	vocabReverse map[int]string // ID → token string
	mergeRanks   map[string]int // "first second" → rank
	addedTokens  map[string]int // special tokens
	specialIDs   map[int]bool

	bosToken  int32
	eosTokens []int32
	unkToken  int32

	addBosToken      bool
	useSentencePiece bool

	pretokenizePattern *regexp.Regexp
	useLookaheadSplit  bool

	byteEncoder map[byte]string
	byteDecoder map[string]byte
}

// LoadHFTokenizer loads a HuggingFace tokenizer from a model directory.
func LoadHFTokenizer(modelDir string) (*HFTokenizer, error) {
	tokPath := filepath.Join(modelDir, "tokenizer.json")
	data, err := os.ReadFile(tokPath)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer.json: %w", err)
	}

	var config tokenizerJSONFormat
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}

	if config.Model.Type != "BPE" {
		return nil, fmt.Errorf("unsupported tokenizer type: %s (only BPE supported)", config.Model.Type)
	}

	// Read optional tokenizer_config.json.
	var tokConfig struct {
		AddBosToken bool            `json:"add_bos_token"`
		EOSToken    json.RawMessage `json:"eos_token"`
		BOSToken    json.RawMessage `json:"bos_token"`
	}
	if cfgData, err := os.ReadFile(filepath.Join(modelDir, "tokenizer_config.json")); err == nil {
		json.Unmarshal(cfgData, &tokConfig)
	}

	// Build vocab.
	vocab := make(map[string]int, len(config.Model.Vocab))
	vocabReverse := make(map[int]string, len(config.Model.Vocab))
	for token, id := range config.Model.Vocab {
		vocab[token] = id
		vocabReverse[id] = token
	}

	// Parse merges.
	var mergeStrings []string
	if err := json.Unmarshal(config.Model.Merges, &mergeStrings); err != nil {
		// Try array-of-arrays format.
		var pairs [][]string
		if err2 := json.Unmarshal(config.Model.Merges, &pairs); err2 != nil {
			return nil, fmt.Errorf("parse merges: %w", err)
		}
		for _, p := range pairs {
			if len(p) == 2 {
				mergeStrings = append(mergeStrings, p[0]+" "+p[1])
			}
		}
	}
	mergeRanks := make(map[string]int, len(mergeStrings))
	for i, m := range mergeStrings {
		mergeRanks[m] = i
	}

	// Handle added/special tokens.
	addedTokens := make(map[string]int)
	specialIDs := make(map[int]bool)
	var bosToken, primaryEOS, unkToken int32 = -1, -1, -1

	configEOS := jsonTokenString(tokConfig.EOSToken)
	configBOS := jsonTokenString(tokConfig.BOSToken)

	for _, tok := range config.AddedTokens {
		addedTokens[tok.Content] = tok.ID
		vocabReverse[tok.ID] = tok.Content
		if tok.Special {
			specialIDs[tok.ID] = true
		}
		c := tok.Content
		if configEOS != "" && c == configEOS {
			primaryEOS = int32(tok.ID)
		} else if configEOS == "" && (c == "<|end_of_text|>" || c == "</s>" || c == "<|endoftext|>" || c == "<|im_end|>") {
			primaryEOS = int32(tok.ID)
		}
		if configBOS != "" && c == configBOS {
			bosToken = int32(tok.ID)
		} else if configBOS == "" && (c == "<|begin_of_text|>" || c == "<s>" || c == "<|startoftext|>") {
			bosToken = int32(tok.ID)
		}
		if c == "<unk>" || c == "<|unk|>" {
			unkToken = int32(tok.ID)
		}
	}
	if bosToken == -1 {
		bosToken = 1
	}
	if primaryEOS == -1 {
		primaryEOS = 2
	}

	// EOS from config.json (authoritative for multi-EOS models).
	var eosTokens []int32
	if cfgData, err := os.ReadFile(filepath.Join(modelDir, "config.json")); err == nil {
		var cfgJSON map[string]any
		if json.Unmarshal(cfgData, &cfgJSON) == nil {
			if v, ok := cfgJSON["eos_token_id"]; ok {
				switch ev := v.(type) {
				case float64:
					eosTokens = []int32{int32(ev)}
				case []any:
					for _, item := range ev {
						if f, ok := item.(float64); ok {
							eosTokens = append(eosTokens, int32(f))
						}
					}
				}
			}
		}
	}
	if len(eosTokens) == 0 {
		eosTokens = []int32{primaryEOS}
	}

	// Check for SentencePiece normalizer.
	useSP := false
	if len(config.Normalizer) > 0 {
		s := string(config.Normalizer)
		useSP = strings.Contains(s, `"Prepend"`) && strings.Contains(s, "\u2581")
	}

	// Pre-tokenization regex.
	pretokRegex := `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`
	useLookahead := false
	if config.PreTokenizer != nil {
		if r := extractSplitRegexFromPT(config.PreTokenizer); r != "" {
			pretokRegex, useLookahead = convertPCREToGo(r)
		}
	}

	byteEnc, byteDec := bytesToUnicodeMap()

	return &HFTokenizer{
		vocab:              vocab,
		vocabReverse:       vocabReverse,
		mergeRanks:         mergeRanks,
		addedTokens:        addedTokens,
		specialIDs:         specialIDs,
		bosToken:           bosToken,
		eosTokens:          eosTokens,
		unkToken:           unkToken,
		addBosToken:        tokConfig.AddBosToken,
		useSentencePiece:   useSP,
		pretokenizePattern: regexp.MustCompile(pretokRegex),
		useLookaheadSplit:  useLookahead,
		byteEncoder:        byteEnc,
		byteDecoder:        byteDec,
	}, nil
}

// SetModelID is a no-op kept for API compatibility. Encoding is now fully Go-native.
func (t *HFTokenizer) SetModelID(string) {}

// VocabSize returns the vocabulary size.
func (t *HFTokenizer) VocabSize() int {
	return len(t.vocab) + len(t.addedTokens)
}

// EOSToken returns the primary EOS token ID.
func (t *HFTokenizer) EOSToken() int32 {
	if len(t.eosTokens) > 0 {
		return t.eosTokens[0]
	}
	return 2
}

// EOSTokenIDs returns all EOS token IDs.
func (t *HFTokenizer) EOSTokenIDs() []int32 { return t.eosTokens }

// BOSToken returns the BOS token ID.
func (t *HFTokenizer) BOSToken() int32 { return t.bosToken }

// Encode converts text to token IDs using byte-level BPE. Fully Go-native.
func (t *HFTokenizer) Encode(text string) ([]int32, error) {
	if text == "" {
		return nil, nil
	}

	var ids []int32
	if t.addBosToken && t.bosToken >= 0 {
		ids = append(ids, t.bosToken)
	}

	if t.useSentencePiece {
		text = "▁" + strings.ReplaceAll(text, " ", "▁")
	}

	// Split on special tokens first.
	parts := t.splitSpecialTokens(text)

	for _, part := range parts {
		if id, ok := t.addedTokens[part]; ok {
			ids = append(ids, int32(id))
			continue
		}

		if t.useSentencePiece {
			ids = append(ids, t.encodeSentencePiece(part)...)
			continue
		}

		// GPT-2 style: pretokenize → bytes → BPE.
		pretokens := t.pretokenize(part)
		for _, pt := range pretokens {
			if pt == "" {
				continue
			}
			tokens := make([]string, len(pt))
			for i, b := range []byte(pt) {
				tokens[i] = t.byteEncoder[b]
			}
			tokens = t.bpe(tokens)
			for _, tok := range tokens {
				if id, ok := t.vocab[tok]; ok {
					ids = append(ids, int32(id))
				} else if t.unkToken >= 0 {
					ids = append(ids, t.unkToken)
				}
			}
		}
	}

	return ids, nil
}

// Decode converts token IDs back to text.
func (t *HFTokenizer) Decode(tokens []int32) string {
	var b strings.Builder
	for _, id := range tokens {
		tok, ok := t.vocabReverse[int(id)]
		if !ok {
			continue
		}
		b.WriteString(t.decodeToken(tok))
	}
	return b.String()
}

// DecodeToken converts a single token ID to its string representation.
func (t *HFTokenizer) DecodeToken(id int) string {
	if t == nil {
		return ""
	}
	tok, ok := t.vocabReverse[id]
	if !ok {
		return ""
	}
	return t.decodeToken(tok)
}

// DecodeTokens converts multiple token IDs to a single string.
func (t *HFTokenizer) DecodeTokens(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		b.WriteString(t.DecodeToken(id))
	}
	return b.String()
}

func (t *HFTokenizer) decodeToken(tok string) string {
	// SentencePiece byte fallback: <0xHH>
	if len(tok) == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>' {
		var b byte
		fmt.Sscanf(tok[3:5], "%02x", &b)
		return string([]byte{b})
	}
	// SentencePiece space: ▁ → space
	if strings.HasPrefix(tok, "▁") {
		return " " + tok[len("▁"):]
	}
	// GPT-2 byte-to-unicode decoding.
	if len(t.byteDecoder) > 0 {
		var bytes []byte
		for _, r := range tok {
			if b, ok := t.byteDecoder[string(r)]; ok {
				bytes = append(bytes, b)
			} else {
				bytes = append(bytes, []byte(string(r))...)
			}
		}
		return string(bytes)
	}
	return tok
}

// --- BPE core ---

func (t *HFTokenizer) bpe(tokens []string) []string {
	if len(tokens) <= 1 {
		return tokens
	}
	for {
		minRank, minPos := -1, -1
		for i := 0; i < len(tokens)-1; i++ {
			if rank, ok := t.mergeRanks[tokens[i]+" "+tokens[i+1]]; ok {
				if minRank == -1 || rank < minRank {
					minRank = rank
					minPos = i
				}
			}
		}
		if minPos == -1 {
			break
		}
		out := make([]string, 0, len(tokens)-1)
		for i := 0; i < len(tokens); i++ {
			if i == minPos {
				out = append(out, tokens[i]+tokens[i+1])
				i++
			} else {
				out = append(out, tokens[i])
			}
		}
		tokens = out
	}
	return tokens
}

func (t *HFTokenizer) encodeSentencePiece(text string) []int32 {
	if id, ok := t.vocab[text]; ok {
		return []int32{int32(id)}
	}
	runes := []rune(text)
	tokens := make([]string, len(runes))
	for i, r := range runes {
		tokens[i] = string(r)
	}
	tokens = t.bpe(tokens)
	var ids []int32
	for _, tok := range tokens {
		if id, ok := t.vocab[tok]; ok {
			ids = append(ids, int32(id))
		} else if t.unkToken >= 0 {
			ids = append(ids, t.unkToken)
		}
	}
	return ids
}

func (t *HFTokenizer) splitSpecialTokens(text string) []string {
	if len(t.addedTokens) == 0 {
		return []string{text}
	}
	type match struct {
		pos   int
		token string
	}
	var matches []match
	for tok := range t.addedTokens {
		start := 0
		for {
			idx := strings.Index(text[start:], tok)
			if idx == -1 {
				break
			}
			matches = append(matches, match{start + idx, tok})
			start += idx + len(tok)
		}
	}
	if len(matches) == 0 {
		return []string{text}
	}
	// Sort by position.
	for i := 0; i < len(matches)-1; i++ {
		for j := 0; j < len(matches)-i-1; j++ {
			if matches[j].pos > matches[j+1].pos {
				matches[j], matches[j+1] = matches[j+1], matches[j]
			}
		}
	}
	var parts []string
	last := 0
	for _, m := range matches {
		if m.pos > last {
			parts = append(parts, text[last:m.pos])
		}
		parts = append(parts, m.token)
		last = m.pos + len(m.token)
	}
	if last < len(text) {
		parts = append(parts, text[last:])
	}
	return parts
}

func (t *HFTokenizer) pretokenize(text string) []string {
	if t.pretokenizePattern == nil {
		return strings.Fields(text)
	}
	matches := t.pretokenizePattern.FindAllString(text, -1)
	if !t.useLookaheadSplit || len(matches) == 0 {
		return matches
	}
	result := make([]string, 0, len(matches))
	for i, m := range matches {
		if i+1 < len(matches) && isSpacesOnly(m) && len(m) > 1 {
			next := matches[i+1]
			if len(next) > 0 && !isAllWS(next) {
				result = append(result, m[:len(m)-1])
				matches[i+1] = m[len(m)-1:] + next
				continue
			}
		}
		result = append(result, m)
	}
	return result
}

func isSpacesOnly(s string) bool {
	for _, r := range s {
		if r != ' ' && r != '\t' {
			return false
		}
	}
	return len(s) > 0
}

func isAllWS(s string) bool {
	for _, r := range s {
		if r != ' ' && r != '\t' && r != '\n' && r != '\r' {
			return false
		}
	}
	return len(s) > 0
}

// --- JSON format types ---

type tokenizerJSONFormat struct {
	Model struct {
		Type   string          `json:"type"`
		Vocab  map[string]int  `json:"vocab"`
		Merges json.RawMessage `json:"merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	Normalizer   json.RawMessage  `json:"normalizer,omitempty"`
	PreTokenizer *preTokenizerFmt `json:"pre_tokenizer,omitempty"`
}

type preTokenizerFmt struct {
	Type          string            `json:"type"`
	Pattern       patternFmt        `json:"pattern,omitempty"`
	Pretokenizers []preTokenizerFmt `json:"pretokenizers,omitempty"`
}

type patternFmt struct {
	value   string
	isRegex bool
}

func (p *patternFmt) UnmarshalJSON(data []byte) error {
	var str string
	if json.Unmarshal(data, &str) == nil {
		p.value = str
		return nil
	}
	var obj struct {
		String string `json:"String"`
		Regex  string `json:"Regex"`
	}
	if json.Unmarshal(data, &obj) == nil {
		if obj.Regex != "" {
			p.value = obj.Regex
			p.isRegex = true
		} else {
			p.value = obj.String
		}
	}
	return nil
}

func extractSplitRegexFromPT(pt *preTokenizerFmt) string {
	if pt == nil {
		return ""
	}
	if pt.Type == "Split" && pt.Pattern.isRegex && pt.Pattern.value != "" {
		return pt.Pattern.value
	}
	if pt.Type == "Sequence" {
		for i := range pt.Pretokenizers {
			if r := extractSplitRegexFromPT(&pt.Pretokenizers[i]); r != "" {
				return r
			}
		}
	}
	return ""
}

func convertPCREToGo(pattern string) (string, bool) {
	hasLookahead := strings.Contains(pattern, `(?!`)
	var result strings.Builder
	i := 0
	for i < len(pattern) {
		if i+3 < len(pattern) && pattern[i] == '(' && pattern[i+1] == '?' && pattern[i+2] == '!' {
			cur := result.String()
			if strings.HasSuffix(cur, `\s+`) {
				result.Reset()
				result.WriteString(cur[:len(cur)-3])
			}
			depth := 1
			j := i + 3
			for j < len(pattern) && depth > 0 {
				if pattern[j] == '\\' && j+1 < len(pattern) {
					j++
				} else if pattern[j] == '(' {
					depth++
				} else if pattern[j] == ')' {
					depth--
				}
				j++
			}
			if j < len(pattern) && pattern[j] == '|' {
				j++
			}
			i = j
			continue
		}
		result.WriteByte(pattern[i])
		i++
	}
	return result.String(), hasLookahead
}

func jsonTokenString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var obj struct{ Content string }
	if json.Unmarshal(raw, &obj) == nil {
		return obj.Content
	}
	return ""
}

// bytesToUnicodeMap creates the GPT-2 byte-to-unicode mapping.
func bytesToUnicodeMap() (map[byte]string, map[string]byte) {
	enc := make(map[byte]string)
	dec := make(map[string]byte)

	var bs []byte
	var cs []rune

	for i := 33; i < 127; i++ {
		bs = append(bs, byte(i))
		cs = append(cs, rune(i))
	}
	for i := 161; i < 173; i++ {
		bs = append(bs, byte(i))
		cs = append(cs, rune(i))
	}
	for i := 174; i < 256; i++ {
		bs = append(bs, byte(i))
		cs = append(cs, rune(i))
	}

	n := 0
	for i := 0; i < 256; i++ {
		found := false
		for _, b := range bs {
			if b == byte(i) {
				found = true
				break
			}
		}
		if !found {
			bs = append(bs, byte(i))
			cs = append(cs, rune(256+n))
			n++
		}
	}

	for i, b := range bs {
		s := string(cs[i])
		enc[b] = s
		dec[s] = b
	}
	return enc, dec
}
