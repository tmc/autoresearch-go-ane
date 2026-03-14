package stories

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// Tokenizer decodes token IDs from llama2.c-compatible tokenizer.bin files.
type Tokenizer struct {
	vocab []string
}

// LoadTokenizer loads tokenizer metadata from tokenizer.bin.
func LoadTokenizer(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var maxTokenLen int32
	if err := binary.Read(f, binary.LittleEndian, &maxTokenLen); err != nil {
		return nil, fmt.Errorf("read tokenizer header: %w", err)
	}
	if maxTokenLen <= 0 {
		return nil, fmt.Errorf("bad tokenizer header: max token len=%d", maxTokenLen)
	}

	vocab := make([]string, Vocab)
	for i := 0; i < Vocab; i++ {
		var score float32
		if err := binary.Read(f, binary.LittleEndian, &score); err != nil {
			if err == io.EOF {
				return nil, fmt.Errorf("unexpected EOF at token %d", i)
			}
			return nil, fmt.Errorf("read token[%d] score: %w", i, err)
		}
		var n int32
		if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
			return nil, fmt.Errorf("read token[%d] length: %w", i, err)
		}
		if n < 0 || n > maxTokenLen*8 {
			return nil, fmt.Errorf("bad token[%d] length=%d", i, n)
		}
		b := make([]byte, n)
		if _, err := io.ReadFull(f, b); err != nil {
			return nil, fmt.Errorf("read token[%d] bytes: %w", i, err)
		}
		vocab[i] = string(b)
	}

	return &Tokenizer{vocab: vocab}, nil
}

// Decode converts a token ID to a text piece.
func (t *Tokenizer) Decode(id int) string {
	if t == nil || id < 0 || id >= len(t.vocab) {
		return ""
	}
	s := t.vocab[id]
	if len(s) >= 5 && s[:3] == "<0x" && s[len(s)-1] == '>' {
		if r, ok := decodeHexTokenByte(s); ok {
			return string([]byte{r})
		}
	}
	return s
}

func decodeHexTokenByte(s string) (byte, bool) {
	if len(s) != 6 { // <0xHH>
		return 0, false
	}
	hi, ok := fromHexNibble(s[3])
	if !ok {
		return 0, false
	}
	lo, ok := fromHexNibble(s[4])
	if !ok {
		return 0, false
	}
	return byte((hi << 4) | lo), true
}

func fromHexNibble(c byte) (byte, bool) {
	switch {
	case c >= '0' && c <= '9':
		return c - '0', true
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10, true
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10, true
	default:
		return 0, false
	}
}
