//go:build !darwin

package stories

func matMulVocabSeqAccelerate(logits, embed, x []float32, vocab, dim, seq int) bool {
	return false
}

func matMulEmbedTAccelerateScale(dx, embed, dLogits []float32, vocab, dim, seq int, scale float32) bool {
	return false
}

func matMulGradEmbedAccelerateScale(dEmbed, dLogits, x []float32, vocab, dim, seq int, scale float32) bool {
	return false
}
