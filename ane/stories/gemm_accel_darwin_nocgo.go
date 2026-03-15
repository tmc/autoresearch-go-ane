//go:build darwin && !cgo

package stories

func matMulVocabSeqAccelerate(logits, embed, x []float32, vocab, dim, seq int) bool {
	return false
}

func matMulEmbedTAccelerate(dx, embed, dLogits []float32, vocab, dim, seq int) bool {
	return false
}

func matMulGradEmbedAccelerate(dEmbed, dLogits, x []float32, vocab, dim, seq int) bool {
	return false
}
