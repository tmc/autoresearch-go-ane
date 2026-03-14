//go:build !darwin

package iosurface

import (
	"fmt"
	"unsafe"
)

type Surface struct{}

func Create(int) (*Surface, error)     { return nil, fmt.Errorf("iosurface requires darwin") }
func (s *Surface) Bytes() int          { return 0 }
func (s *Surface) Ref() unsafe.Pointer { return nil }
func (s *Surface) Write([]byte) error  { return fmt.Errorf("iosurface requires darwin") }
func (s *Surface) Read([]byte) error   { return fmt.Errorf("iosurface requires darwin") }
func (s *Surface) WriteF32([]float32) error {
	return fmt.Errorf("iosurface requires darwin")
}
func (s *Surface) ReadF32([]float32) error {
	return fmt.Errorf("iosurface requires darwin")
}
func (s *Surface) Close() {}
