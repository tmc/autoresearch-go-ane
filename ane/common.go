package ane

import (
	"math"
	"runtime"
)

// --- grad task concurrency ---

const defaultGradTaskConcurrency = 4

var gradTaskLimit = defaultGradTaskConcurrency

// SetGradTaskConcurrency sets the maximum number of concurrent gradient tasks.
//
// Values <= 0 restore the default.
func SetGradTaskConcurrency(n int) {
	if n <= 0 {
		gradTaskLimit = defaultGradTaskConcurrency
		return
	}
	gradTaskLimit = n
}

func gradTaskConcurrency() int {
	n := runtime.GOMAXPROCS(0)
	if n < 1 {
		n = 1
	}
	if n > gradTaskLimit {
		n = gradTaskLimit
	}
	return n
}

// --- offload tile ranges ---

type tileRange struct {
	start int
	size  int
}

func classifierTileRanges(vocab, tile int) []tileRange {
	if vocab <= 0 || tile <= 0 {
		return nil
	}
	n := (vocab + tile - 1) / tile
	ranges := make([]tileRange, 0, n)
	for start := 0; start < vocab; start += tile {
		size := tile
		if rem := vocab - start; rem < size {
			size = rem
		}
		ranges = append(ranges, tileRange{start: start, size: size})
	}
	return ranges
}

// --- forward common (CPU fallback) ---

func causalAttentionCF(out, qf, kf, vf []float32, heads, headDim, seq int) {
	scores := make([]float32, seq)
	probs := make([]float32, seq)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for h := 0; h < heads; h++ {
		base := h * headDim
		for t := 0; t < seq; t++ {
			for j := 0; j < seq; j++ {
				if j > t {
					scores[j] = -65504
					continue
				}
				sum := float32(0)
				for i := 0; i < headDim; i++ {
					sum += qf[(base+i)*seq+t] * kf[(base+i)*seq+j]
				}
				scores[j] = sum * scale
			}
			softmaxRow(probs, scores)
			for i := 0; i < headDim; i++ {
				sum := float32(0)
				for j := 0; j < seq; j++ {
					sum += probs[j] * vf[(base+i)*seq+j]
				}
				out[(base+i)*seq+t] = sum
			}
		}
	}
}

func linearCF(out, weights, in []float32, outCh, inCh, seq int) {
	if linearCFAccelerate(out, weights, in, outCh, inCh, seq) {
		return
	}
	for o := 0; o < outCh; o++ {
		row := weights[o*inCh : (o+1)*inCh]
		for t := 0; t < seq; t++ {
			sum := float32(0)
			for i := 0; i < inCh; i++ {
				sum += row[i] * in[i*seq+t]
			}
			out[o*seq+t] = sum
		}
	}
}

func rmsNormCF(out, x, w []float32, dim, seq int) {
	rmsNormCFWithRRMS(out, nil, x, w, dim, seq)
}

func rmsNormCFWithRRMS(out, rrms, x, w []float32, dim, seq int) {
	parallelForCF(seq, func(start, end int) {
		for t := start; t < end; t++ {
			sum := 0.0
			for i := 0; i < dim; i++ {
				v := float64(x[i*seq+t])
				sum += v * v
			}
			scale := float32(1.0 / math.Sqrt(sum/float64(dim)+1e-5))
			if rrms != nil {
				rrms[t] = scale
			}
			for i := 0; i < dim; i++ {
				out[i*seq+t] = x[i*seq+t] * scale * w[i]
			}
		}
	})
}

func rmsNormRRMS(rrms, x []float32, dim, seq int) {
	if len(rrms) < seq {
		return
	}
	parallelForCF(seq, func(start, end int) {
		for t := start; t < end; t++ {
			sum := 0.0
			for i := 0; i < dim; i++ {
				v := float64(x[i*seq+t])
				sum += v * v
			}
			rrms[t] = float32(1.0 / math.Sqrt(sum/float64(dim)+1e-5))
		}
	})
}

func reluSquared32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	return x * x
}

func rmsNormNoWeightCF(out, x []float32, dim, seq int) {
	rmsNormNoWeightCFWithRRMS(out, nil, x, dim, seq)
}

func rmsNormNoWeightCFWithRRMS(out, rrms, x []float32, dim, seq int) {
	parallelForCF(seq, func(start, end int) {
		for t := start; t < end; t++ {
			sum := 0.0
			for i := 0; i < dim; i++ {
				v := float64(x[i*seq+t])
				sum += v * v
			}
			scale := float32(1.0 / math.Sqrt(sum/float64(dim)+1e-5))
			if rrms != nil {
				rrms[t] = scale
			}
			for i := 0; i < dim; i++ {
				out[i*seq+t] = x[i*seq+t] * scale
			}
		}
	})
}

func rmsNormNoWeightBackwardCF(dx, dy, x []float32, dim, seq int) {
	parallelForCF(seq, func(start, end int) {
		invD := 1.0 / float64(dim)
		for t := start; t < end; t++ {
			sum := 0.0
			for i := 0; i < dim; i++ {
				v := float64(x[i*seq+t])
				sum += v * v
			}
			rrms := 1.0 / math.Sqrt(sum*invD+1e-5)
			dot := 0.0
			for i := 0; i < dim; i++ {
				dot += float64(dy[i*seq+t]) * float64(x[i*seq+t])
			}
			coeff := dot * rrms * rrms * rrms * invD
			for i := 0; i < dim; i++ {
				idx := i*seq + t
				dx[idx] = float32(float64(dy[idx])*rrms - coeff*float64(x[idx]))
			}
		}
	})
}

func qkNormCF(q, k []float32, dim, heads, seq int) {
	headDim := dim / heads
	const qkScale = 1.2
	for h := 0; h < heads; h++ {
		base := h * headDim
		for t := 0; t < seq; t++ {
			// Q head norm
			sumQ := 0.0
			for i := 0; i < headDim; i++ {
				v := float64(q[(base+i)*seq+t])
				sumQ += v * v
			}
			scaleQ := float32(qkScale / math.Sqrt(sumQ/float64(headDim)+1e-5))
			for i := 0; i < headDim; i++ {
				q[(base+i)*seq+t] *= scaleQ
			}
			// K head norm
			sumK := 0.0
			for i := 0; i < headDim; i++ {
				v := float64(k[(base+i)*seq+t])
				sumK += v * v
			}
			scaleK := float32(qkScale / math.Sqrt(sumK/float64(headDim)+1e-5))
			for i := 0; i < headDim; i++ {
				k[(base+i)*seq+t] *= scaleK
			}
		}
	}
}

func qkNormBackwardCF(dq, dk, qPre, kPre []float32, dim, heads, seq int) {
	headDim := dim / heads
	const qkScale = 1.2
	invHD := 1.0 / float64(headDim)
	for h := 0; h < heads; h++ {
		base := h * headDim
		for t := 0; t < seq; t++ {
			// Q backward
			sumQ := 0.0
			for i := 0; i < headDim; i++ {
				v := float64(qPre[(base+i)*seq+t])
				sumQ += v * v
			}
			rrmsQ := 1.0 / math.Sqrt(sumQ*invHD+1e-5)
			dotQ := 0.0
			for i := 0; i < headDim; i++ {
				dotQ += float64(dq[(base+i)*seq+t]) * float64(qPre[(base+i)*seq+t])
			}
			coeffQ := dotQ * rrmsQ * rrmsQ * rrmsQ * invHD * qkScale * qkScale
			for i := 0; i < headDim; i++ {
				idx := (base + i) * seq + t
				dq[idx] = float32(float64(dq[idx])*rrmsQ*qkScale - coeffQ*float64(qPre[idx]))
			}
			// K backward
			sumK := 0.0
			for i := 0; i < headDim; i++ {
				v := float64(kPre[(base+i)*seq+t])
				sumK += v * v
			}
			rrmsK := 1.0 / math.Sqrt(sumK*invHD+1e-5)
			dotK := 0.0
			for i := 0; i < headDim; i++ {
				dotK += float64(dk[(base+i)*seq+t]) * float64(kPre[(base+i)*seq+t])
			}
			coeffK := dotK * rrmsK * rrmsK * rrmsK * invHD * qkScale * qkScale
			for i := 0; i < headDim; i++ {
				idx := (base + i) * seq + t
				dk[idx] = float32(float64(dk[idx])*rrmsK*qkScale - coeffK*float64(kPre[idx]))
			}
		}
	}
}

func logitSoftcap(logits []float32) {
	const cap = 15.0
	const invCap = 1.0 / cap
	for i, v := range logits {
		logits[i] = cap * float32(math.Tanh(float64(v)*invCap))
	}
}

func logitSoftcapBackward(grad, preSoftcap []float32) {
	const cap = 15.0
	const invCap = 1.0 / cap
	for i, g := range grad {
		t := math.Tanh(float64(preSoftcap[i]) * invCap)
		grad[i] = g * float32(1-t*t)
	}
}

func onesSlice(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = 1.0
	}
	return s
}

func softmaxRow(out, in []float32) {
	softmaxRowAccel(out, in)
}

func sigmoid32(x float32) float32 {
	return 1.0 / (1 + float32(math.Exp(float64(-x))))
}

func silu32(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

// smearForwardCF applies bigram mixing in-place (eval path, no save).
// x is channel-first [dim, seq]. smearGate is [dim*dim] row-major.
//   gatePre[d,t] = (smearGate @ x)[d,t]
//   shifted[d,t] = x[d,t-1] (zero for t=0)
//   x[d,t] += smearLambda * sigmoid(gatePre[d,t]) * shifted[d,t]
func smearForwardCF(x, smearGate []float32, smearLambda float32, dim, seq int) {
	gatePre := make([]float32, dim*seq)
	linearCF(gatePre, smearGate, x, dim, dim, seq)
	for d := 0; d < dim; d++ {
		for t := seq - 1; t >= 1; t-- {
			g := sigmoid32(gatePre[d*seq+t])
			x[d*seq+t] += smearLambda * g * x[d*seq+(t-1)]
		}
		// t=0: shifted is zero, no-op.
	}
}

// smearForwardWithSaveCF applies bigram mixing with saved intermediates for backward.
// xPre is the input before smear (saved for backward). x is modified in-place.
func smearForwardWithSaveCF(x, xPre, smearGate []float32, smearLambda float32,
	gatePre, shifted, gateAct []float32, dim, seq int) {
	copy(xPre, x)
	linearCF(gatePre, smearGate, xPre, dim, dim, seq)
	for d := 0; d < dim; d++ {
		for t := 0; t < seq; t++ {
			gateAct[d*seq+t] = sigmoid32(gatePre[d*seq+t])
			if t == 0 {
				shifted[d*seq+t] = 0
			} else {
				shifted[d*seq+t] = xPre[d*seq+(t-1)]
			}
			x[d*seq+t] += smearLambda * gateAct[d*seq+t] * shifted[d*seq+t]
		}
	}
}

// smearBackwardCF computes gradients through smear.
// dOut is dL/d(post-smear x), modified in-place to become dL/d(pre-smear x).
func smearBackwardCF(dOut, xPre, smearGate []float32, smearLambda float32,
	gatePre, shifted, gateAct []float32,
	gSmearGate, gSmearLambda []float32, dim, seq int) {
	// Gradient w.r.t. smearLambda (scalar).
	var dLambda float64
	// dGatePre accumulates gradient through sigmoid gate for smearGate weight update.
	dGatePre := make([]float32, dim*seq)

	for d := 0; d < dim; d++ {
		for t := 0; t < seq; t++ {
			g := gateAct[d*seq+t]
			s := shifted[d*seq+t]
			dy := dOut[d*seq+t]
			contrib := smearLambda * g * s
			_ = contrib
			// dL/dSmearLambda += dy * g * s
			dLambda += float64(dy * g * s)
			// dL/dGatePre[d,t] = dy * smearLambda * s * g*(1-g)
			dGatePre[d*seq+t] = dy * smearLambda * s * g * (1 - g)
			// dL/dShifted[d,t] = dy * smearLambda * g
			dShifted := dy * smearLambda * g
			// shifted[d,t] = xPre[d,t-1], so propagate to dOut[d,t-1]
			if t > 0 {
				dOut[d*seq+(t-1)] += dShifted
			}
		}
	}
	gSmearLambda[0] += float32(dLambda)

	// dL/dSmearGate: gatePre = SmearGate @ xPre, so dSmearGate += dGatePre @ xPre^T
	accumLinearGradCF(gSmearGate, dGatePre, xPre, dim, dim, seq)

	// dL/dxPre also gets contribution from gatePre = SmearGate @ xPre
	// dL/dxPre += SmearGate^T @ dGatePre
	dxFromGate := make([]float32, dim*seq)
	linearBackwardDXCF(dxFromGate, smearGate, dGatePre, dim, dim, seq)
	for i := range dOut {
		dOut[i] += dxFromGate[i]
	}
}

// backoutForwardCF subtracts the scaled mid-layer residual from final hidden.
func backoutForwardCF(finalHidden, xMid []float32, backoutLambda float32, dim, seq int) {
	for i := 0; i < dim*seq; i++ {
		finalHidden[i] -= backoutLambda * xMid[i]
	}
}

// backoutBackwardCF computes gradients through backout.
// dOut is dL/d(post-backout), xMid is the mid-layer snapshot.
func backoutBackwardCF(dOut, xMid []float32, backoutLambda float32,
	gBackoutLambda []float32, dim, seq int) {
	// dL/dBackoutLambda = sum(-dOut[j] * xMid[j])
	var dLambda float64
	for i := 0; i < dim*seq; i++ {
		dLambda -= float64(dOut[i] * xMid[i])
	}
	gBackoutLambda[0] += float32(dLambda)
	// dL/dXMid_from_backout = -backoutLambda * dOut (propagated in layer backward)
	// dL/d(pre-backout finalHidden) = dOut (unchanged, identity)
}
