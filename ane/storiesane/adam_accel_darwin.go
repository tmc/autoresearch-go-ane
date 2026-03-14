//go:build darwin && cgo

package storiesane

/*
#cgo darwin CFLAGS: -O3
#include <stddef.h>
#include <math.h>

static void adam_update_f32(
	float* w,
	const float* g,
	float* m,
	float* v,
	float b1,
	float b2,
	float oneMinusB1,
	float oneMinusB2,
	float invBC1,
	float invBC2,
	float lr,
	float eps,
	float wd,
	size_t n
) {
	for (size_t i = 0; i < n; i++) {
		float gi = g[i];
		float mi = b1 * m[i] + oneMinusB1 * gi;
		float vi = b2 * v[i] + oneMinusB2 * gi * gi;
		m[i] = mi;
		v[i] = vi;
		float mh = mi * invBC1;
		float vh = vi * invBC2;
		float upd = mh / (sqrtf(vh) + eps);
		if (wd != 0.0f) {
			upd += wd * w[i];
		}
		w[i] -= lr * upd;
	}
}
*/
import "C"

func adamUpdateCFAccelerateChunk(w, g, m, v []float32, b1, b2, invBC1, invBC2, lr, eps, wd float32) bool {
	if len(w) == 0 {
		return true
	}
	if len(g) < len(w) || len(m) < len(w) || len(v) < len(w) {
		return false
	}
	C.adam_update_f32(
		(*C.float)(&w[0]),
		(*C.float)(&g[0]),
		(*C.float)(&m[0]),
		(*C.float)(&v[0]),
		C.float(b1),
		C.float(b2),
		C.float(1-b1),
		C.float(1-b2),
		C.float(invBC1),
		C.float(invBC2),
		C.float(lr),
		C.float(eps),
		C.float(wd),
		C.size_t(len(w)),
	)
	return true
}
