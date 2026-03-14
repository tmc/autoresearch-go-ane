//go:build !darwin

package storiesane

import "fmt"

type layerBackward struct{}

func (lb *layerBackward) close() {}

func (e *Engine) ensureBackward() error {
	return fmt.Errorf("ane hybrid backward is unavailable on this platform")
}

func (e *Engine) disableHybridBackward(error) {}
