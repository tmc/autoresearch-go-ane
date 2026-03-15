//go:build darwin

package stories

import "github.com/tmc/apple/accelerate"

func matMulVocabSeqAccelerate(logits, embed, x []float32, vocab, dim, seq int) bool {
	if vocab <= 0 || dim <= 0 || seq <= 0 {
		return false
	}
	if len(logits) < vocab*seq || len(embed) < vocab*dim || len(x) < dim*seq {
		return false
	}
	// logits = embed @ x  (vocab x dim) @ (dim x seq) = (vocab x seq)
	accelerate.Cblas_sgemm(
		accelerate.CblasRowMajor,
		accelerate.CblasNoTrans,
		accelerate.CblasNoTrans,
		vocab, seq, dim,
		1.0, embed, dim,
		x, seq,
		0.0, logits, seq,
	)
	return true
}

func matMulEmbedTAccelerateScale(dx, embed, dLogits []float32, vocab, dim, seq int, scale float32) bool {
	if vocab <= 0 || dim <= 0 || seq <= 0 {
		return false
	}
	if len(dx) < dim*seq || len(embed) < vocab*dim || len(dLogits) < vocab*seq {
		return false
	}
	// dx = embed^T @ dLogits * scale  (dim x vocab) @ (vocab x seq) = (dim x seq)
	accelerate.Cblas_sgemm(
		accelerate.CblasRowMajor,
		accelerate.CblasTrans,
		accelerate.CblasNoTrans,
		dim, seq, vocab,
		scale, embed, dim,
		dLogits, seq,
		0.0, dx, seq,
	)
	return true
}

func matMulGradEmbedAccelerateScale(dEmbed, dLogits, x []float32, vocab, dim, seq int, scale float32) bool {
	if vocab <= 0 || dim <= 0 || seq <= 0 {
		return false
	}
	if len(dEmbed) < vocab*dim || len(dLogits) < vocab*seq || len(x) < dim*seq {
		return false
	}
	// dEmbed = dLogits @ x^T * scale  (vocab x seq) @ (seq x dim) = (vocab x dim)
	accelerate.Cblas_sgemm(
		accelerate.CblasRowMajor,
		accelerate.CblasNoTrans,
		accelerate.CblasTrans,
		vocab, dim, seq,
		scale, dLogits, seq,
		x, seq,
		0.0, dEmbed, dim,
	)
	return true
}
