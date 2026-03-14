//go:build darwin

package storiesane

import (
	"fmt"
	"sync"

	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

var (
	directSequenceProbeMu    sync.Mutex
	directSequenceProbeCache = make(map[int]string)

	probeDirectSequenceCompileFunc = compileDirectSequenceProbe
)

// ProbeDirectSequence reports whether the direct-Go in-memory ANE compile path
// can compile a minimal stories-shaped kernel for seq on this host.
func ProbeDirectSequence(seq int) error {
	if seq <= 0 {
		seq = stories.SeqDefault
	}

	directSequenceProbeMu.Lock()
	if msg, ok := directSequenceProbeCache[seq]; ok {
		directSequenceProbeMu.Unlock()
		if msg == "" {
			return nil
		}
		return fmt.Errorf("%s", msg)
	}
	directSequenceProbeMu.Unlock()

	err := probeDirectSequenceCompileFunc(seq)
	msg := ""
	if err != nil {
		msg = err.Error()
	}

	directSequenceProbeMu.Lock()
	directSequenceProbeCache[seq] = msg
	directSequenceProbeMu.Unlock()

	if msg == "" {
		return nil
	}
	return fmt.Errorf("%s", msg)
}

func compileDirectSequenceProbe(seq int) error {
	layer := stories.LayerWeights{
		Wq:     make([]float32, stories.WQSize),
		Wk:     make([]float32, stories.WQSize),
		Wv:     make([]float32, stories.WQSize),
		Wo:     make([]float32, stories.WOSize),
		W1:     make([]float32, stories.W1Size),
		W2:     make([]float32, stories.W2Size),
		W3:     make([]float32, stories.W3Size),
		RMSAtt: make([]float32, stories.Dim),
		RMSFFN: make([]float32, stories.Dim),
	}
	lf, err := compileStoriesLayerForwardDynamic(layer, seq)
	if err != nil {
		return fmt.Errorf("compile direct forward layer: %w", err)
	}
	defer lf.close()
	lb, err := compileStoriesLayerBackwardDynamic(layer, seq)
	if err != nil {
		return fmt.Errorf("compile direct backward layer: %w", err)
	}
	defer lb.close()
	off := newOffload(&stories.ModelWeights{
		RMSFinal: make([]float32, stories.Dim),
		Embed:    make([]float32, stories.Vocab*stories.Dim),
	}, seq, true, true)
	if off != nil {
		defer off.close()
	}
	return nil
}
