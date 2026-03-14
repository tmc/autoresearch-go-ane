//go:build !darwin

package mil

import (
	"fmt"
	"math"
)

// GenConv reproduces the single-op conv MIL generator from training/ane_mil_gen.h.
func GenConv(inCh, outCh, spatial int) string {
	return fmt.Sprintf(
		"program(1.3)\n"+
			"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"+
			"{\n"+
			"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"+
			"        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"+
			"        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"+
			"        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"+
			"        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"+
			"        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"+
			"        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"+
			"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"+
			"        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"+
			"        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"+
			"        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"+
			"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"+
			"    } -> (y);\n"+
			"}\n",
		inCh, spatial,
		inCh, spatial,
		outCh, inCh, outCh, inCh,
		outCh, spatial,
		outCh, spatial,
	)
}

// GenConvFP16 generates a minimal fp16 conv graph matching the working training path.
func GenConvFP16(inCh, outCh, spatial int) string {
	return fmt.Sprintf(
		"program(1.3)\n"+
			"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"+
			"{\n"+
			"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"+
			"        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"+
			"        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"+
			"        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"+
			"        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"+
			"        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"+
			"        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"+
			"        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x)[name = string(\"conv\")];\n"+
			"    } -> (y);\n"+
			"}\n",
		inCh, spatial,
		outCh, inCh, outCh, inCh,
		outCh, spatial,
	)
}

// GenConvFP32 generates a minimal fp32 conv graph matching the upstream helper.
func GenConvFP32(inCh, outCh, spatial int) string {
	return GenConv(inCh, outCh, spatial)
}

// GenConvDynamicFP16 generates an fp16 conv graph with runtime-provided weights.
func GenConvDynamicFP16(inCh, outCh, spatial int) string {
	return fmt.Sprintf(
		"program(1.3)\n"+
			"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"+
			"{\n"+
			"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [%d, %d, 1, 1]> W) {\n"+
			"        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"+
			"        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"+
			"        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"+
			"        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"+
			"        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"+
			"        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x)[name = string(\"conv\")];\n"+
			"    } -> (y);\n"+
			"}\n",
		inCh, spatial, outCh, inCh, outCh, spatial,
	)
}

// GenMatmul generates a weight-input matmul MIL useful for smoke tests.
func GenMatmul(inCh, outCh, spatial int) string {
	return fmt.Sprintf(
		"program(1.3)\n"+
			"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"+
			"{\n"+
			"    func main<ios18>(tensor<fp32, [1, %d, %d]> x, tensor<fp32, [1, %d, %d]> W) {\n"+
			"        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"+
			"        tensor<fp16, [1, %d, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_x\")];\n"+
			"        tensor<fp16, [1, %d, %d]> W16 = cast(dtype = to_fp16, x = W)[name = string(\"cast_W\")];\n"+
			"        bool tx = const()[name = string(\"tx\"), val = bool(false)];\n"+
			"        bool ty = const()[name = string(\"ty\"), val = bool(false)];\n"+
			"        tensor<fp16, [1, %d, %d]> y16 = matmul(transpose_x = tx, transpose_y = ty, x = W16, y = x16)[name = string(\"mm\")];\n"+
			"        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"+
			"        tensor<fp32, [1, %d, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"+
			"    } -> (y);\n"+
			"}\n",
		inCh, spatial, outCh, inCh,
		inCh, spatial, outCh, inCh,
		outCh, spatial, outCh, spatial,
	)
}

func GenIdentity(channels, spatial int) string {
	return GenConv(channels, channels, spatial)
}

// BuildWeightBlob reproduces training/ane_mil_gen.h weight blob layout.
func BuildWeightBlob(weights []float32, outCh, inCh int) ([]byte, error) {
	want := outCh * inCh
	if len(weights) != want {
		return nil, fmt.Errorf("weight count=%d want=%d", len(weights), want)
	}
	wsize := want * 2
	total := 64 + 64 + wsize
	buf := make([]byte, total)
	buf[0] = 0x01
	buf[4] = 0x02
	chunk := buf[64:]
	chunk[0], chunk[1], chunk[2], chunk[3] = 0xEF, 0xBE, 0xAD, 0xDE
	chunk[4] = 0x01
	putU32(chunk[8:], uint32(wsize))
	putU32(chunk[16:], 128)
	for i, f := range weights {
		h := float32ToHalfBits(f)
		buf[128+i*2] = byte(h)
		buf[128+i*2+1] = byte(h >> 8)
	}
	return buf, nil
}

func BuildIdentityWeightBlob(channels int) ([]byte, error) {
	weights := make([]float32, channels*channels)
	for i := 0; i < channels; i++ {
		weights[i*channels+i] = 1
	}
	return BuildWeightBlob(weights, channels, channels)
}

func putU32(dst []byte, v uint32) {
	dst[0] = byte(v)
	dst[1] = byte(v >> 8)
	dst[2] = byte(v >> 16)
	dst[3] = byte(v >> 24)
}

// float32ToHalfBits converts float32 to IEEE 754 half-precision bits.
func float32ToHalfBits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff

	switch {
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		half := uint16(mant >> shift)
		if (mant>>(shift-1))&1 != 0 {
			half++
		}
		return sign | half
	case exp >= 31:
		if mant == 0 {
			return sign | 0x7c00
		}
		return sign | 0x7c00 | uint16(mant>>13)
	default:
		half := sign | uint16(exp<<10) | uint16(mant>>13)
		if (mant & 0x1000) != 0 {
			half++
		}
		return half
	}
}

// Float32ToHalfBits converts float32 to IEEE 754 half-precision bits.
func Float32ToHalfBits(f float32) uint16 { return float32ToHalfBits(f) }

func Float32ToFP16(f float32) uint16 { return float32ToHalfBits(f) }

// HalfBitsToFloat32 converts IEEE 754 half-precision bits to float32.
func HalfBitsToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x03ff)

	var bits uint32
	switch exp {
	case 0:
		if mant == 0 {
			bits = sign << 31
		} else {
			// Subnormal half -> normalized float
			e := int32(-14)
			m := mant
			for (m & 0x0400) == 0 {
				m <<= 1
				e--
			}
			m &= 0x03ff
			bits = (sign << 31) | (uint32(e+127) << 23) | (m << 13)
		}
	case 0x1f:
		bits = (sign << 31) | 0x7f800000 | (mant << 13)
	default:
		e := int32(exp) - 15 + 127
		bits = (sign << 31) | (uint32(e) << 23) | (mant << 13)
	}
	return math.Float32frombits(bits)
}

func FP16ToFloat32(h uint16) float32 { return HalfBitsToFloat32(h) }
