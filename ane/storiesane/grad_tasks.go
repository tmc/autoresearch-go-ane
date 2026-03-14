package storiesane

import "runtime"

const defaultGradTaskConcurrency = 4

var gradTaskLimit = defaultGradTaskConcurrency

// SetGradTaskConcurrency sets the maximum number of concurrent gradient tasks.
//
// Values <= 0 restore the default.
func SetGradTaskConcurrency(n int) {
	if n <= 0 {
		gradTaskLimit = defaultGradTaskConcurrency
		return
	}
	gradTaskLimit = n
}

func gradTaskConcurrency() int {
	n := runtime.GOMAXPROCS(0)
	if n < 1 {
		n = 1
	}
	if n > gradTaskLimit {
		n = gradTaskLimit
	}
	return n
}
