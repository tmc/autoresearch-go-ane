//go:build darwin

package ane

import (
	"context"
	"fmt"

	aneruntime "github.com/tmc/autoresearch-go-ane/ane/runtime"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/private/appleneuralengine"
)

type objcRuntime struct {
}

func newRuntime() Runtime {
	return &objcRuntime{}
}

func (r *objcRuntime) Probe(context.Context) (ProbeReport, error) {
	if err := aneruntime.EnsureLoaded(); err != nil {
		return ProbeReport{}, err
	}

	class, className := getDeviceInfoClass()
	if class == 0 {
		return ProbeReport{}, fmt.Errorf("ane device info class not found")
	}
	device := appleneuralengine.GetANEDeviceInfoClass()

	report := ProbeReport{
		Available:       true,
		DeviceInfoClass: className,
	}

	report.HasANE = device.HasANE()
	report.IsVirtualMachine = device.IsVirtualMachine()
	report.NumANECores = device.NumANECores()
	report.NumANEs = device.NumANEs()
	report.Architecture = objc.IDToString(device.AneArchitectureType().GetID())
	report.BuildVersion = objc.IDToString(device.BuildVersion().GetID())

	if status, ok := queryConnectStatus(); ok {
		report.ConnectAttempt = true
		report.ConnectStatus = status
	}

	return report, nil
}

func getDeviceInfoClass() (objc.Class, string) {
	name, class := aneruntime.FirstClass("_ANEDeviceInfo", "ANEDeviceInfo")
	return class, name
}

func queryConnectStatus() (uint32, bool) {
	vcClass := appleneuralengine.GetANEVirtualClientClass()
	client := vcClass.Alloc()
	if client.GetID() == 0 {
		return 0, false
	}

	client = client.InitWithSingletonAccess()
	if client.GetID() == 0 {
		client = appleneuralengine.NewANEVirtualClient()
	}
	if client.GetID() == 0 {
		return 0, false
	}
	defer client.Release()
	return client.Connect(), true
}
