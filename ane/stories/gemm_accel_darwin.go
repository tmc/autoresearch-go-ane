//go:build darwin && cgo

package stories

/*
#cgo darwin CFLAGS: -Wno-deprecated-declarations
#cgo darwin LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"

import "sync"

func matMulVocabSeqAccelerate(logits, embed, x []float32, vocab, dim, seq int) bool {
	if vocab <= 0 || dim <= 0 || seq <= 0 {
		return false
	}
	if len(logits) < vocab*seq || len(embed) < vocab*dim || len(x) < dim*seq {
		return false
	}
	// Split vocab into chunks and compute concurrently.
	const nSplit = 24
	chunk := vocab / nSplit
	if chunk > 0 && vocab > 1000 {
		var wg sync.WaitGroup
		for s := 0; s < nSplit; s++ {
			start := s * chunk
			size := chunk
			if s == nSplit-1 {
				size = vocab - start
			}
			wg.Add(1)
			go func(start, size int) {
				defer wg.Done()
				C.cblas_sgemm(
					C.CblasRowMajor, C.CblasNoTrans, C.CblasNoTrans,
					C.int(size), C.int(seq), C.int(dim),
					C.float(1.0),
					(*C.float)(&embed[start*dim]), C.int(dim),
					(*C.float)(&x[0]), C.int(seq),
					C.float(0.0),
					(*C.float)(&logits[start*seq]), C.int(seq),
				)
			}(start, size)
		}
		wg.Wait()
		return true
	}
	C.cblas_sgemm(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.CblasNoTrans,
		C.int(vocab),
		C.int(seq),
		C.int(dim),
		C.float(1.0),
		(*C.float)(&embed[0]),
		C.int(dim),
		(*C.float)(&x[0]),
		C.int(seq),
		C.float(0.0),
		(*C.float)(&logits[0]),
		C.int(seq),
	)
	return true
}

func matMulEmbedTAccelerate(dx, embed, dLogits []float32, vocab, dim, seq int) bool {
	return matMulEmbedTAccelerateScale(dx, embed, dLogits, vocab, dim, seq, 1.0)
}

func matMulEmbedTAccelerateScale(dx, embed, dLogits []float32, vocab, dim, seq int, scale float32) bool {
	if vocab <= 0 || dim <= 0 || seq <= 0 {
		return false
	}
	if len(dx) < dim*seq || len(embed) < vocab*dim || len(dLogits) < vocab*seq {
		return false
	}
	C.cblas_sgemm(
		C.CblasRowMajor,
		C.CblasTrans,
		C.CblasNoTrans,
		C.int(dim),
		C.int(seq),
		C.int(vocab),
		C.float(scale),
		(*C.float)(&embed[0]),
		C.int(dim),
		(*C.float)(&dLogits[0]),
		C.int(seq),
		C.float(0.0),
		(*C.float)(&dx[0]),
		C.int(seq),
	)
	return true
}

func matMulGradEmbedAccelerate(dEmbed, dLogits, x []float32, vocab, dim, seq int) bool {
	return matMulGradEmbedAccelerateScale(dEmbed, dLogits, x, vocab, dim, seq, 1.0)
}

func matMulGradEmbedAccelerateScale(dEmbed, dLogits, x []float32, vocab, dim, seq int, scale float32) bool {
	if vocab <= 0 || dim <= 0 || seq <= 0 {
		return false
	}
	if len(dEmbed) < vocab*dim || len(dLogits) < vocab*seq || len(x) < dim*seq {
		return false
	}
	C.cblas_sgemm(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.CblasTrans,
		C.int(vocab),
		C.int(dim),
		C.int(seq),
		C.float(scale),
		(*C.float)(&dLogits[0]),
		C.int(seq),
		(*C.float)(&x[0]),
		C.int(seq),
		C.float(0.0),
		(*C.float)(&dEmbed[0]),
		C.int(dim),
	)
	return true
}
