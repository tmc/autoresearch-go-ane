package runtime

import (
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
	"github.com/tmc/apple/objc"
)

const frameworkPath = "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"
const coreMLFrameworkPath = "/System/Library/Frameworks/CoreML.framework/CoreML"
const espressoFrameworkPath = "/System/Library/PrivateFrameworks/Espresso.framework/Espresso"

var (
	loadOnce      sync.Once
	loadErr       error
	coreMLOnce    sync.Once
	coreMLLoadErr error
	espressoOnce  sync.Once
	espressoErr   error
)

// EnsureLoaded loads the AppleNeuralEngine private framework once.
func EnsureLoaded() error {
	loadOnce.Do(func() {
		_, err := purego.Dlopen(frameworkPath, purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if err != nil {
			loadErr = fmt.Errorf("load AppleNeuralEngine framework: %w", err)
		}
	})
	return loadErr
}

// EnsureCoreMLLoaded loads the CoreML framework once.
func EnsureCoreMLLoaded() error {
	coreMLOnce.Do(func() {
		_, err := purego.Dlopen(coreMLFrameworkPath, purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if err != nil {
			coreMLLoadErr = fmt.Errorf("load CoreML framework: %w", err)
		}
	})
	return coreMLLoadErr
}

// Deprecated: new code should prefer github.com/tmc/apple/x/espresso.
// EnsureEspressoLoaded loads the Espresso private framework once.
func EnsureEspressoLoaded() error {
	espressoOnce.Do(func() {
		_, err := purego.Dlopen(espressoFrameworkPath, purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if err != nil {
			espressoErr = fmt.Errorf("load Espresso framework: %w", err)
		}
	})
	return espressoErr
}

// FirstClass returns the first available Objective-C class name and handle.
func FirstClass(candidates ...string) (name string, class objc.Class) {
	for _, n := range candidates {
		if c := objc.GetClass(n); c != 0 {
			return n, c
		}
	}
	return "", 0
}
