package storiesane

import (
	"math"

	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

var layerResidualScale = float32(1.0 / math.Sqrt(2.0*float64(stories.NLayers)))

func blendResidualInPlace(sum, base []float32) {
	for i := range sum {
		sum[i] = base[i] + (sum[i]-base[i])*layerResidualScale
	}
}

func addScaledResidual(dst, base, branch []float32) {
	addScaledResidualAccel(dst, base, branch, layerResidualScale)
}

func scaleInto(dst, src []float32, scale float32) {
	scaleIntoAccel(dst, src, scale)
}
