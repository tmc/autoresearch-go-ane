package stories

import (
	"fmt"
	"math"
	"math/rand"
)

const (
	bosToken = 1
	eosToken = 2
)

// DecodeOptions controls autoregressive sampling.
type DecodeOptions struct {
	MaxNewTokens int
	Temperature  float32
	Seed         int64
	PromptTokens []int
}

// DecodeResult contains generated token IDs and decoded text.
type DecodeResult struct {
	Tokens       []int
	PromptLength int
	Text         string
	StoppedOnEOS bool
}

// Decoder runs full Stories110M forward passes with KV cache on CPU.
type Decoder struct {
	w *ModelWeights

	maxSeq int
	hd     int

	kCache []float32 // [layer][pos][dim]
	vCache []float32 // [layer][pos][dim]

	ropeCos []float32 // [pos][hd/2]
	ropeSin []float32 // [pos][hd/2]

	x      []float32
	xn     []float32
	q      []float32
	k      []float32
	v      []float32
	attOut []float32
	x2     []float32
	x2n    []float32
	h1     []float32
	h3     []float32
	ffOut  []float32
	scores []float32
	probs  []float32
	logits []float32

	rng *rand.Rand
}

// ResetCache clears KV cache state.
func (d *Decoder) ResetCache() {
	if d == nil {
		return
	}
	for i := range d.kCache {
		d.kCache[i] = 0
		d.vCache[i] = 0
	}
}

// EvalNextLogits evaluates one token at the given position and returns logits.
//
// The returned slice aliases decoder storage and is overwritten by subsequent calls.
func (d *Decoder) EvalNextLogits(token, pos int) ([]float32, error) {
	return d.forwardOne(token, pos)
}

// NewDecoder builds an inference decoder for the provided model weights.
func NewDecoder(w *ModelWeights, maxSeq int, seed int64) (*Decoder, error) {
	if w == nil {
		return nil, fmt.Errorf("nil model weights")
	}
	if maxSeq <= 0 {
		maxSeq = SeqDefault
	}
	if maxSeq > SeqDefault {
		maxSeq = SeqDefault
	}
	if len(w.Layers) != NLayers {
		return nil, fmt.Errorf("layers=%d want=%d", len(w.Layers), NLayers)
	}
	if len(w.Embed)%Dim != 0 {
		return nil, fmt.Errorf("embed size=%d not divisible by dim=%d", len(w.Embed), Dim)
	}

	hd := Dim / Heads
	ropeCos := make([]float32, maxSeq*(hd/2))
	ropeSin := make([]float32, maxSeq*(hd/2))
	for pos := 0; pos < maxSeq; pos++ {
		for i := 0; i < hd/2; i++ {
			freq := float64(pos) / math.Pow(10000, float64(2*i)/float64(hd))
			idx := pos*(hd/2) + i
			ropeCos[idx] = float32(math.Cos(freq))
			ropeSin[idx] = float32(math.Sin(freq))
		}
	}

	return &Decoder{
		w:       w,
		maxSeq:  maxSeq,
		hd:      hd,
		kCache:  make([]float32, NLayers*maxSeq*Dim),
		vCache:  make([]float32, NLayers*maxSeq*Dim),
		ropeCos: ropeCos,
		ropeSin: ropeSin,
		x:       make([]float32, Dim),
		xn:      make([]float32, Dim),
		q:       make([]float32, Dim),
		k:       make([]float32, Dim),
		v:       make([]float32, Dim),
		attOut:  make([]float32, Dim),
		x2:      make([]float32, Dim),
		x2n:     make([]float32, Dim),
		h1:      make([]float32, Hidden),
		h3:      make([]float32, Hidden),
		ffOut:   make([]float32, Dim),
		scores:  make([]float32, maxSeq),
		probs:   make([]float32, maxSeq),
		logits:  make([]float32, Vocab),
		rng:     rand.New(rand.NewSource(seed)),
	}, nil
}

// Decode runs autoregressive decoding and returns token IDs and decoded text.
func (d *Decoder) Decode(tok *Tokenizer, opts DecodeOptions) (DecodeResult, error) {
	if d == nil {
		return DecodeResult{}, fmt.Errorf("nil decoder")
	}
	maxNew := opts.MaxNewTokens
	if maxNew <= 0 {
		maxNew = 64
	}
	temp := opts.Temperature
	if temp < 0 {
		temp = 0
	}
	if opts.Seed != 0 {
		d.rng = rand.New(rand.NewSource(opts.Seed))
	}

	prompt := append([]int(nil), opts.PromptTokens...)
	if len(prompt) == 0 {
		prompt = []int{bosToken}
	}
	if len(prompt) > d.maxSeq {
		prompt = prompt[len(prompt)-d.maxSeq:]
	}
	for i, t := range prompt {
		if t < 0 || t >= Vocab {
			return DecodeResult{}, fmt.Errorf("prompt token[%d]=%d out of range", i, t)
		}
	}

	for i := range d.kCache {
		d.kCache[i] = 0
		d.vCache[i] = 0
	}

	tokens := append([]int(nil), prompt...)
	for pos := 0; pos < len(prompt)-1; pos++ {
		if _, err := d.forwardOne(prompt[pos], pos); err != nil {
			return DecodeResult{}, err
		}
	}

	generatedText := ""
	stoppedOnEOS := false
	for step := 0; step < maxNew && len(tokens) < d.maxSeq; step++ {
		pos := len(tokens) - 1
		logits, err := d.forwardOne(tokens[pos], pos)
		if err != nil {
			return DecodeResult{}, err
		}
		next := sampleFromLogits(d.rng, logits, temp)
		tokens = append(tokens, next)
		if tok != nil {
			generatedText += tok.Decode(next)
		}
		if next == eosToken {
			stoppedOnEOS = true
			break
		}
	}

	return DecodeResult{
		Tokens:       tokens,
		PromptLength: len(prompt),
		Text:         generatedText,
		StoppedOnEOS: stoppedOnEOS,
	}, nil
}

func (d *Decoder) forwardOne(token, pos int) ([]float32, error) {
	if token < 0 || token >= Vocab {
		return nil, fmt.Errorf("token out of range: %d", token)
	}
	if pos < 0 || pos >= d.maxSeq {
		return nil, fmt.Errorf("position out of range: %d", pos)
	}

	emb := d.w.Embed[token*Dim : (token+1)*Dim]
	copy(d.x, emb)

	for l := 0; l < NLayers; l++ {
		layer := d.w.Layers[l]

		rmsNormVec(d.xn, d.x, layer.RMSAtt)
		matVec(d.q, layer.Wq, d.xn, Dim, Dim)
		matVec(d.k, layer.Wk, d.xn, Dim, Dim)
		matVec(d.v, layer.Wv, d.xn, Dim, Dim)
		d.applyRoPE(pos)
		d.storeKV(l, pos)
		d.attend(l, pos)
		matVec(d.x2, layer.Wo, d.attOut, Dim, Dim)
		for i := 0; i < Dim; i++ {
			d.x2[i] += d.x[i]
		}

		rmsNormVec(d.x2n, d.x2, layer.RMSFFN)
		matVec(d.h1, layer.W1, d.x2n, Hidden, Dim)
		matVec(d.h3, layer.W3, d.x2n, Hidden, Dim)
		for i := 0; i < Hidden; i++ {
			d.h1[i] = silu(d.h1[i]) * d.h3[i]
		}
		matVec(d.ffOut, layer.W2, d.h1, Dim, Hidden)
		for i := 0; i < Dim; i++ {
			d.x[i] = d.x2[i] + d.ffOut[i]
		}
	}

	rmsNormVec(d.xn, d.x, d.w.RMSFinal)
	matVec(d.logits, d.w.Embed, d.xn, Vocab, Dim)
	return d.logits, nil
}

func (d *Decoder) applyRoPE(pos int) {
	stride := d.hd / 2
	for h := 0; h < Heads; h++ {
		base := h * d.hd
		ropeBase := pos * stride
		for i := 0; i < stride; i++ {
			c := d.ropeCos[ropeBase+i]
			s := d.ropeSin[ropeBase+i]
			qi := d.q[base+2*i]
			qj := d.q[base+2*i+1]
			d.q[base+2*i] = qi*c - qj*s
			d.q[base+2*i+1] = qi*s + qj*c
			ki := d.k[base+2*i]
			kj := d.k[base+2*i+1]
			d.k[base+2*i] = ki*c - kj*s
			d.k[base+2*i+1] = ki*s + kj*c
		}
	}
}

func (d *Decoder) storeKV(layer, pos int) {
	base := (layer*d.maxSeq + pos) * Dim
	copy(d.kCache[base:base+Dim], d.k)
	copy(d.vCache[base:base+Dim], d.v)
}

func (d *Decoder) attend(layer, pos int) {
	for i := range d.attOut {
		d.attOut[i] = 0
	}
	invSqrtHD := float32(1.0 / math.Sqrt(float64(d.hd)))

	for h := 0; h < Heads; h++ {
		hBase := h * d.hd
		for t := 0; t <= pos; t++ {
			kBase := (layer*d.maxSeq+t)*Dim + hBase
			dot := float32(0)
			for i := 0; i < d.hd; i++ {
				dot += d.q[hBase+i] * d.kCache[kBase+i]
			}
			d.scores[t] = dot * invSqrtHD
		}
		softmaxPrefix(d.probs, d.scores, pos+1)
		for i := 0; i < d.hd; i++ {
			sum := float32(0)
			for t := 0; t <= pos; t++ {
				vBase := (layer*d.maxSeq+t)*Dim + hBase
				sum += d.probs[t] * d.vCache[vBase+i]
			}
			d.attOut[hBase+i] = sum
		}
	}
}

func rmsNormVec(out, x, w []float32) {
	sum := float64(0)
	for i := 0; i < len(x); i++ {
		v := float64(x[i])
		sum += v * v
	}
	rrms := float32(1.0 / math.Sqrt(sum/float64(len(x))+1e-5))
	for i := 0; i < len(x); i++ {
		out[i] = x[i] * rrms * w[i]
	}
}

func matVec(out, mat, in []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		row := mat[r*cols : (r+1)*cols]
		sum := float32(0)
		for c := 0; c < cols; c++ {
			sum += row[c] * in[c]
		}
		out[r] = sum
	}
}

func silu(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

func softmaxPrefix(dst, src []float32, n int) {
	mx := src[0]
	for i := 1; i < n; i++ {
		if src[i] > mx {
			mx = src[i]
		}
	}
	sum := float64(0)
	for i := 0; i < n; i++ {
		e := float32(math.Exp(float64(src[i] - mx)))
		dst[i] = e
		sum += float64(e)
	}
	inv := float32(1 / sum)
	for i := 0; i < n; i++ {
		dst[i] *= inv
	}
}

func sampleFromLogits(rng *rand.Rand, logits []float32, temperature float32) int {
	if temperature <= 1e-6 {
		return argmax(logits)
	}
	mx := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > mx {
			mx = logits[i]
		}
	}
	invT := float64(1 / temperature)
	total := 0.0
	for i := 0; i < len(logits); i++ {
		total += math.Exp((float64(logits[i]) - float64(mx)) * invT)
	}
	target := rng.Float64() * total
	acc := 0.0
	for i := 0; i < len(logits); i++ {
		acc += math.Exp((float64(logits[i]) - float64(mx)) * invT)
		if acc >= target {
			return i
		}
	}
	return len(logits) - 1
}

func argmax(v []float32) int {
	best := 0
	bestVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > bestVal {
			bestVal = v[i]
			best = i
		}
	}
	return best
}
