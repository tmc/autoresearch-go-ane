package anperf

import (
	"encoding/json"
	"math"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestRingBuffer(t *testing.T) {
	rb := newRingBuffer(5)

	// Empty buffer.
	if snap := rb.snapshot(); snap != nil {
		t.Fatalf("expected nil snapshot, got %d items", len(snap))
	}
	if _, ok := rb.latest(); ok {
		t.Fatal("expected no latest on empty ring")
	}

	// Fill partially.
	for i := 0; i < 3; i++ {
		rb.push(StepResult{Step: i, Loss: float32(i)})
	}
	snap := rb.snapshot()
	if len(snap) != 3 {
		t.Fatalf("expected 3 items, got %d", len(snap))
	}
	for i, s := range snap {
		if s.Step != i {
			t.Errorf("snap[%d].Step = %d, want %d", i, s.Step, i)
		}
	}

	// Overflow.
	for i := 3; i < 8; i++ {
		rb.push(StepResult{Step: i, Loss: float32(i)})
	}
	snap = rb.snapshot()
	if len(snap) != 5 {
		t.Fatalf("expected 5 items, got %d", len(snap))
	}
	// Should contain steps 3..7.
	for i, s := range snap {
		want := i + 3
		if s.Step != want {
			t.Errorf("snap[%d].Step = %d, want %d", i, s.Step, want)
		}
	}

	// Latest.
	latest, ok := rb.latest()
	if !ok || latest.Step != 7 {
		t.Errorf("latest = %v, want step 7", latest)
	}

	// Last N.
	last := rb.last(3)
	if len(last) != 3 {
		t.Fatalf("expected 3 items, got %d", len(last))
	}
	if last[0].Step != 5 || last[2].Step != 7 {
		t.Errorf("last(3) = steps %d..%d, want 5..7", last[0].Step, last[2].Step)
	}
}

func TestRecordAndMetrics(t *testing.T) {
	rec := New(WithCapacity(1000))

	// Record some steps.
	for i := 0; i < 50; i++ {
		rec.Record(StepResult{
			Step:         i,
			Loss:         float32(10.0 - float64(i)*0.1),
			StepMs:       50.0,
			ANEMs:        30.0,
			AdamMs:       10.0,
			CPUMs:        10.0,
			TokensPerSec: 5000.0,
		})
	}

	// Test /api/metrics.
	req := httptest.NewRequest("GET", "/perf/api/metrics", nil)
	w := httptest.NewRecorder()
	handler := rec.Handler("/perf/")
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("metrics status = %d", w.Code)
	}

	var metrics map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &metrics); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if _, ok := metrics["current"]; !ok {
		t.Error("missing 'current' in response")
	}
	if v, ok := metrics["ane_utilization"].(float64); !ok || v < 50 {
		t.Errorf("ane_utilization = %v, expected ~60", v)
	}
}

func TestTimeseries(t *testing.T) {
	rec := New(WithCapacity(1000))
	for i := 0; i < 200; i++ {
		rec.Record(StepResult{Step: i, Loss: float32(10.0 * math.Exp(-float64(i)*0.01))})
	}

	req := httptest.NewRequest("GET", "/perf/api/timeseries?last=50", nil)
	w := httptest.NewRecorder()
	rec.Handler("/perf/").ServeHTTP(w, req)

	var data []StepResult
	if err := json.Unmarshal(w.Body.Bytes(), &data); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if len(data) != 50 {
		t.Errorf("expected 50 items, got %d", len(data))
	}
}

func TestExperiments(t *testing.T) {
	rec := New()

	rec.RecordExperiment("baseline",
		map[string]interface{}{"lr": 3e-4, "seq": 256},
		5.5, 5*time.Minute, 1000, "ok")

	rec.RecordExperiment("lr_sweep",
		map[string]interface{}{"lr": 1e-3, "seq": 256},
		4.8, 5*time.Minute, 1000, "ok")

	req := httptest.NewRequest("GET", "/perf/api/experiments", nil)
	w := httptest.NewRecorder()
	rec.Handler("/perf/").ServeHTTP(w, req)

	var exps []map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &exps); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if len(exps) != 2 {
		t.Fatalf("expected 2 experiments, got %d", len(exps))
	}
	// Second experiment should show improvement.
	imp := exps[1]["improvement_pct"].(float64)
	if imp <= 0 {
		t.Errorf("expected positive improvement, got %v", imp)
	}
}

func TestSystemInfo(t *testing.T) {
	rec := New()
	rec.SetChip("Apple M4 Max")
	rec.SetMemory("128 GB")
	rec.SetANECores(16)
	rec.SetHyperparams(map[string]interface{}{
		"lr":       3e-4,
		"seq":      256,
		"accum":    4,
		"use_ane":  true,
	})

	req := httptest.NewRequest("GET", "/perf/api/system", nil)
	w := httptest.NewRecorder()
	rec.Handler("/perf/").ServeHTTP(w, req)

	var info SystemInfo
	if err := json.Unmarshal(w.Body.Bytes(), &info); err != nil {
		t.Fatalf("json decode: %v", err)
	}
	if info.Chip != "Apple M4 Max" {
		t.Errorf("chip = %q", info.Chip)
	}
	if info.ANECores != 16 {
		t.Errorf("ane_cores = %d", info.ANECores)
	}
}

func TestDownsample(t *testing.T) {
	data := make([]StepResult, 10000)
	for i := range data {
		data[i] = StepResult{Step: i, Loss: float32(math.Sin(float64(i)/100) + 5)}
	}

	ds := downsample(data, 500)
	if len(ds) != 500 {
		t.Fatalf("downsample: expected 500, got %d", len(ds))
	}
	// First and last must be preserved.
	if ds[0].Step != 0 {
		t.Error("first point not preserved")
	}
	if ds[len(ds)-1].Step != 9999 {
		t.Error("last point not preserved")
	}
}

func TestSSEStream(t *testing.T) {
	rec := New()

	// Start SSE in a goroutine; send a few steps; verify data arrives.
	ts := httptest.NewServer(rec.Handler("/"))
	defer ts.Close()

	// Record in background.
	go func() {
		time.Sleep(50 * time.Millisecond)
		for i := 0; i < 5; i++ {
			rec.Record(StepResult{Step: i, Loss: float32(5 - i)})
			time.Sleep(10 * time.Millisecond)
		}
	}()

	// Just verify the endpoint is reachable and returns event-stream.
	client := ts.Client()
	resp, err := client.Get(ts.URL + "/api/stream")
	if err != nil {
		t.Fatalf("GET /api/stream: %v", err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("content-type = %q, want text/event-stream", ct)
	}
}

func TestUIServing(t *testing.T) {
	rec := New()
	ts := httptest.NewServer(rec.Handler("/"))
	defer ts.Close()

	resp, err := ts.Client().Get(ts.URL + "/index.html")
	if err != nil {
		t.Fatalf("GET /index.html: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("status = %d", resp.StatusCode)
	}
}

func BenchmarkRecord(b *testing.B) {
	rec := New(WithCapacity(100000))
	sr := StepResult{
		Step:         0,
		Loss:         5.5,
		StepMs:       50,
		ANEMs:        30,
		AdamMs:       10,
		CPUMs:        10,
		TokensPerSec: 5000,
		Components:   map[string]float64{"compile": 5, "finalHead": 8, "embedGrad": 3},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sr.Step = i
		rec.Record(sr)
	}
}

func BenchmarkSnapshot(b *testing.B) {
	rec := New(WithCapacity(100000))
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 100000; i++ {
		rec.Record(StepResult{Step: i, Loss: float32(rng.Float64() * 10)})
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec.ring.snapshot()
	}
}
