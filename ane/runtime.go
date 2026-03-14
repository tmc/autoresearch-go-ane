package ane

import "context"

// ProbeReport summarizes ANE runtime information discovered via private APIs.
type ProbeReport struct {
	Available        bool
	HasANE           bool
	IsVirtualMachine bool
	NumANECores      uint32
	NumANEs          uint32
	Architecture     string
	BuildVersion     string
	ConnectStatus    uint32
	ConnectAttempt   bool
	DeviceInfoClass  string
}

// Runtime probes ANE capabilities.
type Runtime interface {
	Probe(context.Context) (ProbeReport, error)
}

// New returns a runtime probe implementation for the current platform.
func New() Runtime {
	return newRuntime()
}
