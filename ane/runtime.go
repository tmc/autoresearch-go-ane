package ane

import (
	"fmt"
	"time"
)

// Prepare initializes best-effort ANE components so Diagnostics can report
// actual runtime state before the first step.
func (e *Engine) Prepare() {
	if e == nil {
		return
	}
	e.ensureOffload()
	if e.useANE {
		if err := e.ensureLayers(); err != nil {
			e.disableLayerForward(err)
		}
	}
	if e.hybridBackwardRequested {
		_ = e.ensureBackward()
	}
}

func (e *Engine) refreshANERuntimeForWeights() time.Duration {
	if e == nil || !e.useANE {
		return 0
	}
	start := time.Now()
	offloadStart := time.Now()
	if e.off == nil {
		e.offDirty = true
		e.ensureOffload()
	} else if err := e.off.refreshWeights(e.mw); err != nil {
		e.offDirty = true
		e.ensureOffload()
	}
	e.stepMetrics.addCustomDuration("RefreshOffloadNS", time.Since(offloadStart))
	layerStart := time.Now()
	if err := e.refreshLayerWeights(); err != nil {
		e.stepMetrics.addCustomDuration("RefreshLayerNS", time.Since(layerStart))
		reinitStart := time.Now()
		e.invalidateLayerForward()
		if ensureErr := e.ensureLayers(); ensureErr != nil {
			e.disableLayerForward(ensureErr)
		}
		e.stepMetrics.addCustomDuration("RefreshLayerReinitNS", time.Since(reinitStart))
	} else {
		e.stepMetrics.addCustomDuration("RefreshLayerNS", time.Since(layerStart))
	}
	if e.hybridBackwardRequested {
		backwardStart := time.Now()
		if err := e.refreshBackwardWeights(); err != nil {
			e.stepMetrics.addCustomDuration("RefreshBackwardNS", time.Since(backwardStart))
			reinitStart := time.Now()
			e.invalidateHybridBackward()
			if ensureErr := e.ensureBackward(); ensureErr != nil {
				e.disableHybridBackward(ensureErr)
			}
			e.stepMetrics.addCustomDuration("RefreshBackwardReinitNS", time.Since(reinitStart))
		} else {
			e.stepMetrics.addCustomDuration("RefreshBackwardNS", time.Since(backwardStart))
		}
	}
	dur := time.Since(start)
	e.stepMetrics.addCustomDuration("RefreshTotalNS", dur)
	e.stepMetrics.addRefresh(dur)
	return dur
}

// startAsyncRefresh begins refreshing ANE kernels on a background goroutine.
// The caller must call waitAsyncRefresh before accessing layers or backward
// kernels.
func (e *Engine) startAsyncRefresh() {
	if e == nil || !e.useANE {
		return
	}
	ch := make(chan time.Duration, 1)
	e.asyncRefreshDone = ch
	go func() {
		ch <- e.refreshANERuntimeForWeights()
	}()
}

// waitAsyncRefresh waits for a pending async refresh to complete and returns
// the compile duration. Returns 0 if no async refresh is pending.
func (e *Engine) waitAsyncRefresh() time.Duration {
	if e == nil || e.asyncRefreshDone == nil {
		return 0
	}
	dur := <-e.asyncRefreshDone
	e.asyncRefreshDone = nil
	return dur
}

func (e *Engine) ensureOffload() {
	if e == nil || !e.useANE || e.mw == nil {
		return
	}
	if !e.offDirty && e.off != nil {
		return
	}
	e.off = refreshOffload(e.off, e.mw, e.seq, true, e.cpuClassifierHead)
	e.offDirty = false
}

func (e *Engine) invalidateLayerForward() {
	if e == nil || e.layerInitErr != nil {
		return
	}
	for i := range e.layers {
		if e.layers[i] != nil {
			e.layers[i].close()
		}
	}
	e.layers = nil
	e.layersInit = false
	e.layersDirty = false
}

func (e *Engine) invalidateHybridBackward() {
	if e == nil || e.backwardInitErr != nil {
		return
	}
	for i := range e.backward {
		if e.backward[i] != nil {
			e.backward[i].close()
		}
	}
	e.backward = nil
	e.backwardInit = false
	e.backwardDirty = false
}

func (e *Engine) refreshLayerWeights() error {
	if e == nil || !e.useANE {
		return nil
	}
	if !e.layersInit {
		return e.ensureLayers()
	}
	if e.layerInitErr != nil {
		return e.layerInitErr
	}
	_, err := compileParallel(len(e.layers), func(i int) (struct{}, error) {
		if e.layers[i] == nil {
			return struct{}{}, fmt.Errorf("refresh layer weights: layer %d is nil", i)
		}
		err := e.layers[i].refreshWeights(layerForwardWeights{
			Wq: e.mw.Layers[i].Wq,
			Wk: e.mw.Layers[i].Wk,
			Wv: e.mw.Layers[i].Wv,
			Wo: e.mw.Layers[i].Wo,
			W1: e.mw.Layers[i].W1,
			W2: e.mw.Layers[i].W2,
		})
		return struct{}{}, err
	}, func(struct{}) {})
	return err
}

func (e *Engine) refreshBackwardWeights() error {
	if e == nil || !e.hybridBackwardRequested {
		return nil
	}
	if !e.backwardInit {
		return e.ensureBackward()
	}
	if e.backwardInitErr != nil {
		return e.backwardInitErr
	}
	_, err := compileParallel(len(e.backward), func(i int) (struct{}, error) {
		if e.backward[i] == nil {
			return struct{}{}, fmt.Errorf("refresh backward weights: layer %d is nil", i)
		}
		return struct{}{}, e.backward[i].refreshWeights(e.mw.Layers[i])
	}, func(struct{}) {})
	return err
}
