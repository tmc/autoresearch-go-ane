//go:build darwin

package mil

import (
	"fmt"

	xane "github.com/tmc/apple/x/ane"
	xmil "github.com/tmc/apple/x/ane/mil"
)

func GenConv(inCh, outCh, spatial int) string {
	return xmil.GenConv(inCh, outCh, spatial)
}

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

func GenConvFP32(inCh, outCh, spatial int) string {
	return xmil.GenConvFP32(inCh, outCh, spatial)
}

// GenConvDynamicFP16 keeps the local runtime-weight variant until x/ane/mil
// grows an equivalent helper.
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

func GenMatmul(inCh, outCh, spatial int) string {
	return xmil.GenMatmul(inCh, outCh, spatial)
}

func GenIdentity(channels, spatial int) string {
	return xmil.GenIdentity(channels, spatial)
}

func BuildWeightBlob(weights []float32, outCh, inCh int) ([]byte, error) {
	return xmil.BuildWeightBlob(weights, outCh, inCh)
}

func BuildIdentityWeightBlob(channels int) ([]byte, error) {
	return xmil.BuildIdentityWeightBlob(channels)
}

func Float32ToHalfBits(f float32) uint16 {
	return xane.Float32ToFP16(f)
}

func HalfBitsToFloat32(h uint16) float32 {
	return xane.FP16ToFloat32(h)
}

func Float32ToFP16(f float32) uint16 {
	return xane.Float32ToFP16(f)
}

func FP16ToFloat32(h uint16) float32 {
	return xane.FP16ToFloat32(h)
}
