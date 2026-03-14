//go:build !darwin

package storiesane

import (
	"fmt"

	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

type layerForwardWeights struct {
	RMSAtt []float32
	Wq     []float32
	Wk     []float32
	Wv     []float32
	Wo     []float32
	RMSFFN []float32
	W1     []float32
	W2     []float32
	W3     []float32
}

type layerForward struct{}

var compileStoriesLayerForwardFunc = compileStoriesLayerForward

func compileStoriesLayerForward(stories.LayerWeights, int) (*layerForward, error) {
	return nil, fmt.Errorf("ane layer forward is unavailable on this platform")
}

func (lf *layerForward) close() {}

func (lf *layerForward) run([]float32, []float32) error {
	return fmt.Errorf("ane layer forward is unavailable on this platform")
}

func (lf *layerForward) runWithTaps([]float32, []float32, *layerCache) error {
	return fmt.Errorf("ane layer forward is unavailable on this platform")
}
