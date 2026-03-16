// Package anperf provides real-time performance visualization for ML training
// on Apple Neural Engine hardware. It collects step-level metrics (loss, timing
// breakdowns, throughput) and serves an interactive web dashboard over HTTP.
//
// Quick start:
//
//	perf := anperf.New()
//	http.Handle("/perf/", perf.Handler("/perf/"))
//	go http.ListenAndServe(":9090", nil)
//
//	// In your training loop:
//	perf.Record(anperf.StepResult{
//	    Step:     step,
//	    Loss:     res.Loss,
//	    StepMs:   float64(res.StepDuration) / float64(time.Millisecond),
//	    ANEMs:    float64(res.ANEEvalDuration) / float64(time.Millisecond),
//	    AdamMs:   float64(res.AdamDuration) / float64(time.Millisecond),
//	    CPUMs:    float64(res.CPUWorkDuration) / float64(time.Millisecond),
//	    TokensPerSec: tokensPerSec,
//	})
package anperf

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"net/http"
	"runtime"
	"strings"
	"sync"
	"time"
)

//go:embed ui/*
var uiFS embed.FS

// StepResult captures metrics for a single training step.
type StepResult struct {
	Step         int     `json:"step"`
	Loss         float32 `json:"loss"`
	ValLoss      float32 `json:"val_loss,omitempty"`
	StepMs       float64 `json:"step_ms"`
	ANEMs        float64 `json:"ane_ms"`
	AdamMs       float64 `json:"adam_ms"`
	CPUMs        float64 `json:"cpu_ms"`
	CompileMs    float64 `json:"compile_ms,omitempty"`
	TokensPerSec float64 `json:"tokens_per_sec"`
	LR           float64 `json:"lr,omitempty"`

	// Component-level timing breakdown.
	Components map[string]float64 `json:"components,omitempty"`
}

// ExperimentResult captures the outcome of a training experiment.
type ExperimentResult struct {
	ID          int                    `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Timestamp   time.Time              `json:"timestamp"`
	Params      map[string]interface{} `json:"params"`
	ValLoss     float64                `json:"val_loss"`
	Duration    time.Duration          `json:"duration"`
	Steps       int                    `json:"steps"`
	Status      string                 `json:"status"`
}

// SystemInfo describes the hardware and training configuration.
type SystemInfo struct {
	Chip          string                 `json:"chip"`
	Memory        string                 `json:"memory,omitempty"`
	ANECores      int                    `json:"ane_cores,omitempty"`
	OS            string                 `json:"os"`
	GoVersion     string                 `json:"go_version"`
	StartTime     time.Time              `json:"start_time"`
	Hyperparams   map[string]interface{} `json:"hyperparams,omitempty"`
	TotalSteps    int                    `json:"total_steps"`
	UptimeSeconds float64                `json:"uptime_seconds"`
}

const defaultRingSize = 100_000

// ringBuffer is a fixed-capacity circular buffer for StepResults.
type ringBuffer struct {
	data  []StepResult
	head  int
	count int
	cap   int
}

func newRingBuffer(capacity int) *ringBuffer {
	return &ringBuffer{
		data: make([]StepResult, capacity),
		cap:  capacity,
	}
}

func (rb *ringBuffer) push(sr StepResult) {
	rb.data[rb.head] = sr
	rb.head = (rb.head + 1) % rb.cap
	if rb.count < rb.cap {
		rb.count++
	}
}

// snapshot returns all stored results in chronological order.
func (rb *ringBuffer) snapshot() []StepResult {
	if rb.count == 0 {
		return nil
	}
	out := make([]StepResult, rb.count)
	start := (rb.head - rb.count + rb.cap) % rb.cap
	for i := 0; i < rb.count; i++ {
		out[i] = rb.data[(start+i)%rb.cap]
	}
	return out
}

// last returns the most recent n results in chronological order.
func (rb *ringBuffer) last(n int) []StepResult {
	if n <= 0 || rb.count == 0 {
		return nil
	}
	if n > rb.count {
		n = rb.count
	}
	out := make([]StepResult, n)
	start := (rb.head - n + rb.cap) % rb.cap
	for i := 0; i < n; i++ {
		out[i] = rb.data[(start+i)%rb.cap]
	}
	return out
}

func (rb *ringBuffer) latest() (StepResult, bool) {
	if rb.count == 0 {
		return StepResult{}, false
	}
	idx := (rb.head - 1 + rb.cap) % rb.cap
	return rb.data[idx], true
}

// Recorder collects training metrics and serves the dashboard.
type Recorder struct {
	mu          sync.RWMutex
	ring        *ringBuffer
	experiments []ExperimentResult
	sysInfo     SystemInfo
	startTime   time.Time
	nextExpID   int

	// SSE subscribers.
	ssesMu sync.Mutex
	sses   map[chan StepResult]struct{}
}

// Option configures a Recorder.
type Option func(*Recorder)

// WithCapacity sets the ring buffer capacity (default 100,000 steps).
func WithCapacity(n int) Option {
	return func(r *Recorder) {
		r.ring = newRingBuffer(n)
	}
}

// WithSystemInfo sets initial system information.
func WithSystemInfo(info SystemInfo) Option {
	return func(r *Recorder) {
		r.sysInfo = info
	}
}

// New creates a Recorder with the given options.
func New(opts ...Option) *Recorder {
	r := &Recorder{
		ring:      newRingBuffer(defaultRingSize),
		startTime: time.Now(),
		sses:      make(map[chan StepResult]struct{}),
		sysInfo: SystemInfo{
			Chip:      detectChip(),
			OS:        runtime.GOOS + "/" + runtime.GOARCH,
			GoVersion: runtime.Version(),
			StartTime: time.Now(),
		},
	}
	for _, o := range opts {
		o(r)
	}
	return r
}

// Record stores a step result and notifies all SSE subscribers.
func (r *Recorder) Record(sr StepResult) {
	r.mu.Lock()
	r.ring.push(sr)
	r.sysInfo.TotalSteps = sr.Step
	r.sysInfo.UptimeSeconds = time.Since(r.startTime).Seconds()
	r.mu.Unlock()

	// Non-blocking broadcast to SSE subscribers.
	r.ssesMu.Lock()
	for ch := range r.sses {
		select {
		case ch <- sr:
		default:
			// Drop if subscriber is slow.
		}
	}
	r.ssesMu.Unlock()
}

// RecordExperiment logs a completed experiment.
func (r *Recorder) RecordExperiment(name string, params map[string]interface{}, valLoss float64, duration time.Duration, steps int, status string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.nextExpID++
	exp := ExperimentResult{
		ID:        r.nextExpID,
		Name:      name,
		Timestamp: time.Now(),
		Params:    params,
		ValLoss:   valLoss,
		Duration:  duration,
		Steps:     steps,
		Status:    status,
	}
	r.experiments = append(r.experiments, exp)
}

// SetHyperparams updates the displayed hyperparameters.
func (r *Recorder) SetHyperparams(params map[string]interface{}) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.sysInfo.Hyperparams = params
}

// SetChip overrides auto-detected chip info.
func (r *Recorder) SetChip(chip string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.sysInfo.Chip = chip
}

// SetMemory sets memory info string.
func (r *Recorder) SetMemory(mem string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.sysInfo.Memory = mem
}

// SetANECores sets the number of ANE cores.
func (r *Recorder) SetANECores(n int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.sysInfo.ANECores = n
}

// Handler returns an http.Handler that serves the dashboard at the given prefix.
// Use like: http.Handle("/perf/", perf.Handler("/perf/"))
func (r *Recorder) Handler(prefix string) http.Handler {
	mux := http.NewServeMux()

	// Serve the embedded UI.
	uiSub, _ := fs.Sub(uiFS, "ui")
	fileServer := http.FileServer(http.FS(uiSub))

	mux.HandleFunc(prefix+"api/metrics", r.handleMetrics)
	mux.HandleFunc(prefix+"api/timeseries", r.handleTimeseries)
	mux.HandleFunc(prefix+"api/experiments", r.handleExperiments)
	mux.HandleFunc(prefix+"api/system", r.handleSystem)
	mux.HandleFunc(prefix+"api/stream", r.handleSSE)
	mux.Handle(prefix, http.StripPrefix(prefix, fileServer))

	return mux
}

func (r *Recorder) handleMetrics(w http.ResponseWriter, req *http.Request) {
	r.mu.RLock()
	latest, ok := r.ring.latest()
	recent := r.ring.last(100)
	r.mu.RUnlock()

	if !ok {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"waiting"}`))
		return
	}

	// Compute moving averages from recent data.
	var avgLoss, avgStepMs, avgTokens float64
	var avgANEMs, avgCPUMs, avgAdamMs float64
	n := float64(len(recent))
	for _, s := range recent {
		avgLoss += float64(s.Loss)
		avgStepMs += s.StepMs
		avgTokens += s.TokensPerSec
		avgANEMs += s.ANEMs
		avgCPUMs += s.CPUMs
		avgAdamMs += s.AdamMs
	}
	if n > 0 {
		avgLoss /= n
		avgStepMs /= n
		avgTokens /= n
		avgANEMs /= n
		avgCPUMs /= n
		avgAdamMs /= n
	}

	// Compute ANE utilization as percentage of step time spent on ANE.
	aneUtil := 0.0
	if avgStepMs > 0 {
		aneUtil = (avgANEMs / avgStepMs) * 100
	}

	type metricsResponse struct {
		Current    StepResult `json:"current"`
		AvgLoss    float64    `json:"avg_loss"`
		AvgStepMs  float64    `json:"avg_step_ms"`
		AvgTokens  float64    `json:"avg_tokens_per_sec"`
		ANEUtil    float64    `json:"ane_utilization"`
		AvgANEMs   float64    `json:"avg_ane_ms"`
		AvgCPUMs   float64    `json:"avg_cpu_ms"`
		AvgAdamMs  float64    `json:"avg_adam_ms"`
		TotalSteps int        `json:"total_steps"`
	}

	resp := metricsResponse{
		Current:    latest,
		AvgLoss:    roundTo(avgLoss, 6),
		AvgStepMs:  roundTo(avgStepMs, 2),
		AvgTokens:  roundTo(avgTokens, 1),
		ANEUtil:    roundTo(aneUtil, 1),
		AvgANEMs:   roundTo(avgANEMs, 2),
		AvgCPUMs:   roundTo(avgCPUMs, 2),
		AvgAdamMs:  roundTo(avgAdamMs, 2),
		TotalSteps: latest.Step,
	}

	writeJSON(w, resp)
}

func (r *Recorder) handleTimeseries(w http.ResponseWriter, req *http.Request) {
	// Support ?last=N query param.
	lastN := 0
	if v := req.URL.Query().Get("last"); v != "" {
		fmt.Sscanf(v, "%d", &lastN)
	}

	r.mu.RLock()
	var data []StepResult
	if lastN > 0 {
		data = r.ring.last(lastN)
	} else {
		data = r.ring.snapshot()
	}
	r.mu.RUnlock()

	if data == nil {
		data = []StepResult{}
	}

	// Downsample if there are too many points for the browser.
	maxPoints := 5000
	if v := req.URL.Query().Get("max"); v != "" {
		fmt.Sscanf(v, "%d", &maxPoints)
	}
	if len(data) > maxPoints {
		data = downsample(data, maxPoints)
	}

	writeJSON(w, data)
}

func (r *Recorder) handleExperiments(w http.ResponseWriter, req *http.Request) {
	r.mu.RLock()
	exps := make([]ExperimentResult, len(r.experiments))
	copy(exps, r.experiments)
	r.mu.RUnlock()

	if exps == nil {
		exps = []ExperimentResult{}
	}

	// Add improvement % relative to previous experiment.
	type expWithImprovement struct {
		ExperimentResult
		ImprovementPct float64 `json:"improvement_pct"`
		DurationSecs   float64 `json:"duration_secs"`
	}
	result := make([]expWithImprovement, len(exps))
	for i, e := range exps {
		imp := 0.0
		if i > 0 && exps[i-1].ValLoss > 0 {
			imp = ((exps[i-1].ValLoss - e.ValLoss) / exps[i-1].ValLoss) * 100
		}
		result[i] = expWithImprovement{
			ExperimentResult: e,
			ImprovementPct:   roundTo(imp, 2),
			DurationSecs:     e.Duration.Seconds(),
		}
	}

	writeJSON(w, result)
}

func (r *Recorder) handleSystem(w http.ResponseWriter, req *http.Request) {
	r.mu.RLock()
	info := r.sysInfo
	info.UptimeSeconds = time.Since(r.startTime).Seconds()
	r.mu.RUnlock()

	writeJSON(w, info)
}

// handleSSE serves Server-Sent Events for real-time step updates.
func (r *Recorder) handleSSE(w http.ResponseWriter, req *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	ch := make(chan StepResult, 64)
	r.ssesMu.Lock()
	r.sses[ch] = struct{}{}
	r.ssesMu.Unlock()

	defer func() {
		r.ssesMu.Lock()
		delete(r.sses, ch)
		r.ssesMu.Unlock()
	}()

	// Send initial keepalive.
	fmt.Fprintf(w, ":ok\n\n")
	flusher.Flush()

	ctx := req.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case sr := <-ch:
			data, _ := json.Marshal(sr)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	}
}

// downsample reduces a time series using largest-triangle-three-buckets.
func downsample(data []StepResult, targetSize int) []StepResult {
	if len(data) <= targetSize || targetSize < 3 {
		return data
	}

	out := make([]StepResult, 0, targetSize)
	out = append(out, data[0]) // Always keep first.

	bucketSize := float64(len(data)-2) / float64(targetSize-2)

	for i := 0; i < targetSize-2; i++ {
		bucketStart := int(float64(i)*bucketSize) + 1
		bucketEnd := int(float64(i+1)*bucketSize) + 1
		if bucketEnd > len(data)-1 {
			bucketEnd = len(data) - 1
		}

		// Next bucket average for triangle area calc.
		nextStart := int(float64(i+1)*bucketSize) + 1
		nextEnd := int(float64(i+2)*bucketSize) + 1
		if nextEnd > len(data) {
			nextEnd = len(data)
		}
		var avgX, avgY float64
		for j := nextStart; j < nextEnd; j++ {
			avgX += float64(data[j].Step)
			avgY += float64(data[j].Loss)
		}
		count := float64(nextEnd - nextStart)
		if count > 0 {
			avgX /= count
			avgY /= count
		}

		// Pick point with max triangle area in current bucket.
		bestIdx := bucketStart
		bestArea := -1.0
		prevX := float64(out[len(out)-1].Step)
		prevY := float64(out[len(out)-1].Loss)
		for j := bucketStart; j < bucketEnd; j++ {
			area := math.Abs((prevX-avgX)*(float64(data[j].Loss)-prevY) -
				(prevX-float64(data[j].Step))*(avgY-prevY)) * 0.5
			if area > bestArea {
				bestArea = area
				bestIdx = j
			}
		}
		out = append(out, data[bestIdx])
	}

	out = append(out, data[len(data)-1]) // Always keep last.
	return out
}

func writeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(v)
}

func roundTo(val float64, decimals int) float64 {
	pow := math.Pow(10, float64(decimals))
	return math.Round(val*pow) / pow
}

func detectChip() string {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		return "Apple Silicon (arm64)"
	}
	return strings.ToUpper(runtime.GOARCH)
}
