package stories

const (
	Dim        = 768
	Hidden     = 2048
	Heads      = 12
	SeqDefault = 256
	NLayers    = 12
	Vocab      = 32000
)

const (
	WQSize = Dim * Dim
	WOSize = Dim * Dim
	W1Size = Hidden * Dim
	W2Size = Dim * Hidden
	W3Size = Hidden * Dim
)

type Llama2Config struct {
	Dim       int32
	HiddenDim int32
	NLayers   int32
	NHeads    int32
	NKVHeads  int32
	VocabSize int32
	SeqLen    int32
}
