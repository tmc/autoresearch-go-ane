//go:build darwin

package ane

import (
	"runtime"
	"sync"
)

func parallelForCF(n int, fn func(start, end int)) {
	if n <= 0 {
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || n < workers*4 {
		fn(0, n)
		return
	}
	workers = min(workers, n)
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for worker := range workers {
		start := worker * chunk
		if start >= n {
			break
		}
		end := min(start+chunk, n)
		wg.Go(func() {
			fn(start, end)
		})
	}
	wg.Wait()
}
