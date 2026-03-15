//go:build !darwin

package ane

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

// --- backward ---

type layerBackward struct{}

func (lb *layerBackward) close() {}

func (e *Engine) ensureBackward() error {
	return fmt.Errorf("ane hybrid backward is unavailable on this platform")
}

func (e *Engine) disableHybridBackward(error) {}

// --- layer forward ---

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

// --- offload ---

type offload struct{}

func newOffload(*stories.ModelWeights, int, bool, bool) *offload               { return nil }
func refreshOffload(*offload, *stories.ModelWeights, int, bool, bool) *offload { return nil }
func (o *offload) close()                                                      {}
func (o *offload) hasRMSForward() bool                                         { return false }
func (o *offload) hasClassifierForward() bool                                  { return false }
func (o *offload) hasSoftmax() bool                                            { return false }
func (o *offload) hasClassifierBackward() bool                                 { return false }
func (o *offload) hasRMSBackward() bool                                        { return false }
func (o *offload) disableRMSForward()                                          {}
func (o *offload) disableClassifierForward()                                   {}
func (o *offload) disableSoftmax()                                             {}
func (o *offload) disableClassifierBackward()                                  {}
func (o *offload) disableRMSBackward()                                         {}
func (o *offload) runRMSForward([]float32, []float32) error {
	return nil
}
func (o *offload) runClassifierForward([]float32, []float32) error {
	return nil
}
func (o *offload) runClassifierSoftmax([]float32, []float32) error {
	return nil
}
func (o *offload) runSoftmax([]float32) error {
	return nil
}
func (o *offload) runClassifierBackward([]float32, []float32) error {
	return nil
}
func (o *offload) runRMSBackward([]float32, []float32, []float32) error {
	return nil
}

// --- parallel ---

func parallelForCF(n int, fn func(start, end int)) {
	if n <= 0 {
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || n < workers*4 {
		fn(0, n)
		return
	}
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		start := worker * chunk
		if start >= n {
			break
		}
		end := start + chunk
		if end > n {
			end = n
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			fn(start, end)
		}(start, end)
	}
	wg.Wait()
}

// --- grad tasks ---

type gradTasks struct {
	jobs   chan func()
	tasks  sync.WaitGroup
	worker sync.WaitGroup
}

func newGradTasks() *gradTasks {
	n := gradTaskConcurrency()
	if n <= 0 {
		return nil
	}
	g := &gradTasks{
		jobs: make(chan func(), n*3),
	}
	for i := 0; i < n; i++ {
		g.worker.Add(1)
		go func() {
			defer g.worker.Done()
			for fn := range g.jobs {
				fn()
				g.tasks.Done()
			}
		}()
	}
	return g
}

func (g *gradTasks) Go(fn func()) {
	if g == nil {
		fn()
		return
	}
	g.tasks.Add(1)
	g.jobs <- fn
}

func (g *gradTasks) Wait() {
	if g == nil {
		return
	}
	g.tasks.Wait()
}

func (g *gradTasks) Close() {
	if g == nil {
		return
	}
	close(g.jobs)
	g.worker.Wait()
}
