package storiesane

import (
	"os"
	"runtime"
	"strconv"
	"sync"
)

const defaultMaxCompileConcurrency = 8

var maxCompileConcurrency = configuredMaxCompileConcurrency()

func configuredMaxCompileConcurrency() int {
	s := os.Getenv("ANE_COMPILE_CONCURRENCY")
	if s == "" {
		return defaultMaxCompileConcurrency
	}
	n, err := strconv.Atoi(s)
	if err != nil || n <= 0 {
		return defaultMaxCompileConcurrency
	}
	return n
}

type compileResult[T any] struct {
	index int
	value T
	err   error
}

func compileParallel[T any](n int, compile func(int) (T, error), closeValue func(T)) ([]T, error) {
	values := make([]T, n)
	if n == 0 {
		return values, nil
	}
	workers := compileConcurrency(n)
	if workers == 1 {
		for i := 0; i < n; i++ {
			v, err := compile(i)
			if err != nil {
				closeCompiledValues(values, closeValue)
				closeValue(v)
				return nil, err
			}
			values[i] = v
		}
		return values, nil
	}

	jobs := make(chan int, n)
	results := make(chan compileResult[T], n)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := range jobs {
				v, err := compile(index)
				results <- compileResult[T]{index: index, value: v, err: err}
			}
		}()
	}
	for i := 0; i < n; i++ {
		jobs <- i
	}
	close(jobs)
	go func() {
		wg.Wait()
		close(results)
	}()

	var firstErr error
	for res := range results {
		if res.err != nil {
			if firstErr == nil {
				firstErr = res.err
			}
			closeValue(res.value)
			continue
		}
		values[res.index] = res.value
	}
	if firstErr != nil {
		closeCompiledValues(values, closeValue)
		return nil, firstErr
	}
	return values, nil
}

func compileConcurrency(n int) int {
	if n <= 1 {
		return n
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if workers > maxCompileConcurrency {
		workers = maxCompileConcurrency
	}
	if workers > n {
		workers = n
	}
	return workers
}

func closeCompiledValues[T any](values []T, closeValue func(T)) {
	for _, v := range values {
		closeValue(v)
	}
}
