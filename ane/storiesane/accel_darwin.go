//go:build darwin && cgo

package storiesane

/*
#cgo darwin CFLAGS: -Wno-deprecated-declarations
#cgo darwin LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>

// siluBackward computes dh1[i] = dGate[i] * h3[i] * sig*(1+h1*(1-sig))
// and dh3[i] = dGate[i] * h1[i] * sig
// where sig = 1/(1+exp(-h1[i]))
//
// Processes in chunks to bound stack usage via alloca. Each chunk uses
// vvexpf for the vectorized exp() calls.
static void silu_backward_f32(
	float* restrict dh1,
	float* restrict dh3,
	const float* restrict dGate,
	const float* restrict h1,
	const float* restrict h3,
	int n
) {
	enum { kChunk = 4096 };
	for (int off = 0; off < n; off += kChunk) {
		int cnt = n - off;
		if (cnt > kChunk) cnt = kChunk;
		float negH1[kChunk];
		float minusOne = -1.0f;
		vDSP_vsmul(h1 + off, 1, &minusOne, negH1, 1, cnt);
		float expBuf[kChunk];
		vvexpf(expBuf, negH1, &cnt);
		for (int i = 0; i < cnt; i++) {
			int idx = off + i;
			float sig = 1.0f / (1.0f + expBuf[i]);
			float siluGrad = sig * (1.0f + h1[idx] * (1.0f - sig));
			dh1[idx] = dGate[idx] * h3[idx] * siluGrad;
			dh3[idx] = dGate[idx] * (h1[idx] * sig);
		}
	}
}

// softmax_row_f32 computes numerically-stable softmax over n elements.
static void softmax_row_f32(float* out, const float* in, int n) {
	float maxv;
	vDSP_maxv(in, 1, &maxv, n);
	float negMax = -maxv;
	vDSP_vsadd(in, 1, &negMax, out, 1, n);
	vvexpf(out, out, &n);
	float sum;
	vDSP_sve(out, 1, &sum, n);
	float inv = 1.0f / sum;
	vDSP_vsmul(out, 1, &inv, out, 1, n);
}
*/
import "C"

import "unsafe"

func scaleSliceAccel(v []float32, scale float32) {
	if len(v) == 0 || scale == 1 {
		return
	}
	C.vDSP_vsmul(
		(*C.float)(unsafe.Pointer(&v[0])), 1,
		(*C.float)(unsafe.Pointer(&scale)),
		(*C.float)(unsafe.Pointer(&v[0])), 1,
		C.vDSP_Length(len(v)),
	)
}

func addSliceAccel(dst, src []float32) {
	if len(dst) == 0 {
		return
	}
	C.vDSP_vadd(
		(*C.float)(unsafe.Pointer(&src[0])), 1,
		(*C.float)(unsafe.Pointer(&dst[0])), 1,
		(*C.float)(unsafe.Pointer(&dst[0])), 1,
		C.vDSP_Length(len(dst)),
	)
}

func scaleIntoAccel(dst, src []float32, scale float32) {
	if len(dst) == 0 {
		return
	}
	C.vDSP_vsmul(
		(*C.float)(unsafe.Pointer(&src[0])), 1,
		(*C.float)(unsafe.Pointer(&scale)),
		(*C.float)(unsafe.Pointer(&dst[0])), 1,
		C.vDSP_Length(len(dst)),
	)
}

// addScaledResidualAccel: dst[i] = base[i] + scale*branch[i]
func addScaledResidualAccel(dst, base, branch []float32, scale float32) {
	if len(dst) == 0 {
		return
	}
	// dst = scale * branch
	C.vDSP_vsmul(
		(*C.float)(unsafe.Pointer(&branch[0])), 1,
		(*C.float)(unsafe.Pointer(&scale)),
		(*C.float)(unsafe.Pointer(&dst[0])), 1,
		C.vDSP_Length(len(dst)),
	)
	// dst = dst + base
	C.vDSP_vadd(
		(*C.float)(unsafe.Pointer(&base[0])), 1,
		(*C.float)(unsafe.Pointer(&dst[0])), 1,
		(*C.float)(unsafe.Pointer(&dst[0])), 1,
		C.vDSP_Length(len(dst)),
	)
}

func siluBackwardAccel(dh1, dh3, dGate, h1, h3 []float32) {
	n := len(dh1)
	if n == 0 {
		return
	}
	C.silu_backward_f32(
		(*C.float)(unsafe.Pointer(&dh1[0])),
		(*C.float)(unsafe.Pointer(&dh3[0])),
		(*C.float)(unsafe.Pointer(&dGate[0])),
		(*C.float)(unsafe.Pointer(&h1[0])),
		(*C.float)(unsafe.Pointer(&h3[0])),
		C.int(n),
	)
}

// linearCFAccelerate computes out = weights @ x using BLAS.
// Layout: weights [outCh x inCh] row-major, x [inCh x seq] channel-first,
// out [outCh x seq] channel-first.
func linearCFAccelerate(out, weights, x []float32, outCh, inCh, seq int) bool {
	if outCh <= 0 || inCh <= 0 || seq <= 0 {
		return false
	}
	if len(out) < outCh*seq || len(weights) < outCh*inCh || len(x) < inCh*seq {
		return false
	}
	C.cblas_sgemm(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.CblasNoTrans,
		C.int(outCh),
		C.int(seq),
		C.int(inCh),
		C.float(1.0),
		(*C.float)(unsafe.Pointer(&weights[0])),
		C.int(inCh),
		(*C.float)(unsafe.Pointer(&x[0])),
		C.int(seq),
		C.float(0.0),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(seq),
	)
	return true
}

func softmaxRowAccel(out, in []float32) {
	n := len(in)
	if n == 0 {
		return
	}
	C.softmax_row_f32(
		(*C.float)(unsafe.Pointer(&out[0])),
		(*C.float)(unsafe.Pointer(&in[0])),
		C.int(n),
	)
}

