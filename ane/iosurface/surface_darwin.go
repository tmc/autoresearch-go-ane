//go:build darwin

package iosurface

/*
#cgo darwin LDFLAGS: -framework IOSurface -framework CoreFoundation
#include <IOSurface/IOSurface.h>
#include <CoreFoundation/CoreFoundation.h>
#include <string.h>

static IOSurfaceRef ane_create_surface(size_t bytes) {
	CFNumberRef width = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &bytes);
	size_t one = 1;
	CFNumberRef height = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &one);
	CFNumberRef bpe = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &one);
	CFNumberRef bpr = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &bytes);
	CFNumberRef alloc = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &bytes);
	uint32_t pf = 0;
	CFNumberRef pixel = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &pf);

	const void* keys[] = {
		kIOSurfaceWidth,
		kIOSurfaceHeight,
		kIOSurfaceBytesPerElement,
		kIOSurfaceBytesPerRow,
		kIOSurfaceAllocSize,
		kIOSurfacePixelFormat,
	};
	const void* vals[] = {width, height, bpe, bpr, alloc, pixel};
	CFDictionaryRef dict = CFDictionaryCreate(
		kCFAllocatorDefault,
		keys,
		vals,
		6,
		&kCFTypeDictionaryKeyCallBacks,
		&kCFTypeDictionaryValueCallBacks);
	IOSurfaceRef s = IOSurfaceCreate(dict);
	CFRelease(dict);
	CFRelease(width);
	CFRelease(height);
	CFRelease(bpe);
	CFRelease(bpr);
	CFRelease(alloc);
	CFRelease(pixel);
	return s;
}

static int ane_surface_write(IOSurfaceRef s, const void* p, size_t n) {
	if (s == NULL || p == NULL) return -1;
	if (IOSurfaceLock(s, 0, NULL) != kIOReturnSuccess) return -2;
	void* base = IOSurfaceGetBaseAddress(s);
	if (base == NULL) {
		IOSurfaceUnlock(s, 0, NULL);
		return -3;
	}
	memcpy(base, p, n);
	IOSurfaceUnlock(s, 0, NULL);
	return 0;
}

static int ane_surface_read(IOSurfaceRef s, void* p, size_t n) {
	if (s == NULL || p == NULL) return -1;
	if (IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return -2;
	void* base = IOSurfaceGetBaseAddress(s);
	if (base == NULL) {
		IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
		return -3;
	}
	memcpy(p, base, n);
	IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
	return 0;
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Surface is a minimal IOSurface wrapper used by ANE request I/O plumbing.
type Surface struct {
	ref   C.IOSurfaceRef
	bytes int
}

func Create(bytes int) (*Surface, error) {
	if bytes <= 0 {
		return nil, fmt.Errorf("create IOSurface: bytes must be > 0")
	}
	s := C.ane_create_surface(C.size_t(bytes))
	if s == 0 {
		return nil, fmt.Errorf("create IOSurface: returned nil")
	}
	return &Surface{ref: s, bytes: bytes}, nil
}

func (s *Surface) Bytes() int { return s.bytes }

// Ref returns the underlying IOSurfaceRef as an integer handle.
//
// Returning uintptr avoids leaking cgo pointers into Go heap values.
func (s *Surface) Ref() uintptr {
	if s == nil {
		return 0
	}
	return uintptr(s.ref)
}

func (s *Surface) Write(src []byte) error {
	if s == nil || s.ref == 0 {
		return fmt.Errorf("write IOSurface: nil surface")
	}
	if len(src) != s.bytes {
		return fmt.Errorf("write IOSurface: got %d bytes, want %d", len(src), s.bytes)
	}
	if len(src) == 0 {
		return nil
	}
	st := C.ane_surface_write(s.ref, unsafe.Pointer(&src[0]), C.size_t(len(src)))
	if st != 0 {
		return fmt.Errorf("write IOSurface failed: status=%d", int(st))
	}
	return nil
}

func (s *Surface) Read(dst []byte) error {
	if s == nil || s.ref == 0 {
		return fmt.Errorf("read IOSurface: nil surface")
	}
	if len(dst) != s.bytes {
		return fmt.Errorf("read IOSurface: got %d bytes, want %d", len(dst), s.bytes)
	}
	if len(dst) == 0 {
		return nil
	}
	st := C.ane_surface_read(s.ref, unsafe.Pointer(&dst[0]), C.size_t(len(dst)))
	if st != 0 {
		return fmt.Errorf("read IOSurface failed: status=%d", int(st))
	}
	return nil
}

func (s *Surface) WriteF32(src []float32) error {
	if s == nil || s.ref == 0 {
		return fmt.Errorf("write IOSurface f32: nil surface")
	}
	if len(src)*4 != s.bytes {
		return fmt.Errorf("write IOSurface f32: got %d bytes, want %d", len(src)*4, s.bytes)
	}
	if len(src) == 0 {
		return nil
	}
	if C.IOSurfaceLock(s.ref, 0, nil) != C.kIOReturnSuccess {
		return fmt.Errorf("write IOSurface f32: lock failed")
	}
	base := C.IOSurfaceGetBaseAddress(s.ref)
	if base == nil {
		C.IOSurfaceUnlock(s.ref, 0, nil)
		return fmt.Errorf("write IOSurface f32: nil base address")
	}
	dst := unsafe.Slice((*float32)(base), len(src))
	copy(dst, src)
	C.IOSurfaceUnlock(s.ref, 0, nil)
	return nil
}

func (s *Surface) ReadF32(dst []float32) error {
	if s == nil || s.ref == 0 {
		return fmt.Errorf("read IOSurface f32: nil surface")
	}
	if len(dst)*4 != s.bytes {
		return fmt.Errorf("read IOSurface f32: got %d bytes, want %d", len(dst)*4, s.bytes)
	}
	if len(dst) == 0 {
		return nil
	}
	if C.IOSurfaceLock(s.ref, C.kIOSurfaceLockReadOnly, nil) != C.kIOReturnSuccess {
		return fmt.Errorf("read IOSurface f32: lock failed")
	}
	base := C.IOSurfaceGetBaseAddress(s.ref)
	if base == nil {
		C.IOSurfaceUnlock(s.ref, C.kIOSurfaceLockReadOnly, nil)
		return fmt.Errorf("read IOSurface f32: nil base address")
	}
	src := unsafe.Slice((*float32)(base), len(dst))
	copy(dst, src)
	C.IOSurfaceUnlock(s.ref, C.kIOSurfaceLockReadOnly, nil)
	return nil
}

func (s *Surface) Close() {
	if s == nil || s.ref == 0 {
		return
	}
	C.CFRelease(C.CFTypeRef(s.ref))
	s.ref = 0
}
