// Package ane provides high-level, idiomatic entry points for ANE workloads.
//
// Use OpenEvaluator for synchronous single-model eval from Go:
//
//	ev, err := ane.OpenEvaluator(ane.EvalOptions{
//		ModelPath:   "/path/model.mlmodelc",
//		ModelKey:    "s",
//		InputBytes:  4096,
//		OutputBytes: 4096,
//	})
//	if err != nil {
//		// handle error
//	}
//	defer ev.Close()
//
//	input := make([]float32, 1024)
//	output := make([]float32, 1024)
//	if err := ev.EvalF32(input, output); err != nil {
//		// handle error
//	}
package ane
