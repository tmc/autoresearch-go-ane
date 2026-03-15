package ane

import (
	"maps"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tmc/apple/x/ane/dynamicmatmul"
	"github.com/tmc/apple/x/ane/model"
)

type aneStepMetrics struct {
	aneEvalNS     int64
	finalHeadNS   int64
	embedGradNS   int64
	rmsDWNS       int64
	dwGEMMNS      int64
	dwWaitNS      int64
	adamNS        int64
	refreshNS     int64
	collectCustom bool

	mu      sync.Mutex
	metrics map[string]float64
}

func (m *aneStepMetrics) reset() {
	if m == nil {
		return
	}
	atomic.StoreInt64(&m.aneEvalNS, 0)
	atomic.StoreInt64(&m.finalHeadNS, 0)
	atomic.StoreInt64(&m.embedGradNS, 0)
	atomic.StoreInt64(&m.rmsDWNS, 0)
	atomic.StoreInt64(&m.dwGEMMNS, 0)
	atomic.StoreInt64(&m.dwWaitNS, 0)
	atomic.StoreInt64(&m.adamNS, 0)
	atomic.StoreInt64(&m.refreshNS, 0)
	m.mu.Lock()
	clear(m.metrics)
	m.mu.Unlock()
}

func (m *aneStepMetrics) addHW(ns uint64) {
	if m == nil || ns == 0 {
		return
	}
	atomic.AddInt64(&m.aneEvalNS, int64(ns))
}

func (m *aneStepMetrics) addModelEvalStats(st model.EvalStats) {
	if m == nil {
		return
	}
	if st.HWExecutionNS != 0 {
		atomic.AddInt64(&m.aneEvalNS, int64(st.HWExecutionNS))
	}
	if !m.collectCustom || len(st.Metrics) == 0 {
		return
	}
	m.mu.Lock()
	if m.metrics == nil {
		m.metrics = make(map[string]float64, len(st.Metrics))
	}
	for k, v := range st.Metrics {
		m.metrics[k] += v
	}
	m.mu.Unlock()
}

func (m *aneStepMetrics) addDynamicEvalStats(st dynamicmatmul.EvalStats) {
	if m == nil {
		return
	}
	if st.HWExecutionNS != 0 {
		atomic.AddInt64(&m.aneEvalNS, int64(st.HWExecutionNS))
	}
	if !m.collectCustom || len(st.Metrics) == 0 {
		return
	}
	m.mu.Lock()
	if m.metrics == nil {
		m.metrics = make(map[string]float64, len(st.Metrics))
	}
	for k, v := range st.Metrics {
		m.metrics[k] += v
	}
	m.mu.Unlock()
}

func (m *aneStepMetrics) addFinalHead(d time.Duration) {
	if m == nil || d <= 0 {
		return
	}
	atomic.AddInt64(&m.finalHeadNS, int64(d))
}

func (m *aneStepMetrics) addEmbedGrad(d time.Duration) {
	if m == nil || d <= 0 {
		return
	}
	atomic.AddInt64(&m.embedGradNS, int64(d))
}

func (m *aneStepMetrics) addRMSDW(d time.Duration) {
	if m == nil || d <= 0 {
		return
	}
	atomic.AddInt64(&m.rmsDWNS, int64(d))
}

func (m *aneStepMetrics) addDWGEMM(d time.Duration) {
	if m == nil || d <= 0 {
		return
	}
	atomic.AddInt64(&m.dwGEMMNS, int64(d))
}

func (m *aneStepMetrics) addDWWait(d time.Duration) {
	if m == nil || d <= 0 {
		return
	}
	atomic.AddInt64(&m.dwWaitNS, int64(d))
}

func (m *aneStepMetrics) addAdam(d time.Duration) {
	if m == nil || d <= 0 {
		return
	}
	atomic.AddInt64(&m.adamNS, int64(d))
}

func (m *aneStepMetrics) addRefresh(d time.Duration) {
	if m == nil || d <= 0 {
		return
	}
	atomic.AddInt64(&m.refreshNS, int64(d))
}

func (m *aneStepMetrics) aneEval() time.Duration {
	if m == nil {
		return 0
	}
	return time.Duration(atomic.LoadInt64(&m.aneEvalNS))
}

func (m *aneStepMetrics) finalHead() time.Duration {
	if m == nil {
		return 0
	}
	return time.Duration(atomic.LoadInt64(&m.finalHeadNS))
}

func (m *aneStepMetrics) embedGrad() time.Duration {
	if m == nil {
		return 0
	}
	return time.Duration(atomic.LoadInt64(&m.embedGradNS))
}

func (m *aneStepMetrics) rmsDW() time.Duration {
	if m == nil {
		return 0
	}
	return time.Duration(atomic.LoadInt64(&m.rmsDWNS))
}

func (m *aneStepMetrics) dwGEMM() time.Duration {
	if m == nil {
		return 0
	}
	return time.Duration(atomic.LoadInt64(&m.dwGEMMNS))
}

func (m *aneStepMetrics) dwWait() time.Duration {
	if m == nil {
		return 0
	}
	return time.Duration(atomic.LoadInt64(&m.dwWaitNS))
}

func (m *aneStepMetrics) adam() time.Duration {
	if m == nil {
		return 0
	}
	return time.Duration(atomic.LoadInt64(&m.adamNS))
}

func (m *aneStepMetrics) refresh() time.Duration {
	if m == nil {
		return 0
	}
	return time.Duration(atomic.LoadInt64(&m.refreshNS))
}

func (m *aneStepMetrics) customMetrics() map[string]float64 {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.metrics) == 0 {
		return nil
	}
	return maps.Clone(m.metrics)
}

func (m *aneStepMetrics) addCustomMetric(name string, value float64) {
	if m == nil || !m.collectCustom || name == "" || value == 0 {
		return
	}
	m.mu.Lock()
	if m.metrics == nil {
		m.metrics = make(map[string]float64, 8)
	}
	m.metrics[name] += value
	m.mu.Unlock()
}

func (m *aneStepMetrics) addCustomDuration(name string, d time.Duration) {
	if d <= 0 {
		return
	}
	m.addCustomMetric(name, float64(d))
}

func (m *aneStepMetrics) enableCustomMetrics() {
	if m == nil {
		return
	}
	m.collectCustom = true
}

func (m *aneStepMetrics) wantsCustomMetrics() bool {
	return m != nil && m.collectCustom
}

func evalKernelTracked(metrics *aneStepMetrics, k *model.Kernel) error {
	start := time.Now()
	if metrics == nil {
		return k.Eval()
	}
	if !metrics.wantsCustomMetrics() {
		if err := k.Eval(); err != nil {
			return err
		}
		metrics.addHW(uint64(time.Since(start)))
		return nil
	}
	st, err := k.EvalWithStats()
	if err != nil {
		return err
	}
	metrics.addModelEvalStats(st)
	if st.HWExecutionNS == 0 {
		metrics.addHW(uint64(time.Since(start)))
	}
	return nil
}
