//go:build !darwin

package ane

import (
	"context"
	"fmt"
)

type unsupportedRuntime struct{}

func newRuntime() Runtime {
	return unsupportedRuntime{}
}

func (unsupportedRuntime) Probe(context.Context) (ProbeReport, error) {
	return ProbeReport{}, fmt.Errorf("ane probing requires darwin")
}
