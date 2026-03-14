//go:build !darwin

package storiesane

import "sync"

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
