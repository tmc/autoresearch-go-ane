//go:build darwin && cgo

package ane

/*
#cgo darwin CFLAGS: -Wno-deprecated-declarations
#cgo darwin LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
#include <string.h>

// BNNS-based fp16 weight GEMV: reads fp16 weights, fp32 input/output.
// Creates a filter, applies it, destroys it. For repeated use, cache the filter.
static int bnns_gemv_f16(
	float *out,
	const uint16_t *weights_f16,
	const float *x,
	int outDim, int inDim
) {
	BNNSLayerParametersFullyConnected params;
	memset(&params, 0, sizeof(params));

	params.i_desc.layout = BNNSDataLayoutVector;
	params.i_desc.size[0] = inDim;
	params.i_desc.data_type = BNNSDataTypeFloat32;

	params.o_desc.layout = BNNSDataLayoutVector;
	params.o_desc.size[0] = outDim;
	params.o_desc.data_type = BNNSDataTypeFloat32;

	params.w_desc.layout = BNNSDataLayoutRowMajorMatrix;
	params.w_desc.size[0] = inDim;
	params.w_desc.size[1] = outDim;
	params.w_desc.data_type = BNNSDataTypeFloat16;
	params.w_desc.data = (void *)weights_f16;

	params.bias.data_type = BNNSDataTypeFloat32;
	params.bias.size[0] = 0;
	params.bias.data = NULL;

	params.activation.function = BNNSActivationFunctionIdentity;

	BNNSFilterParameters filterParams;
	memset(&filterParams, 0, sizeof(filterParams));

	BNNSFilter filter = BNNSFilterCreateLayerFullyConnected(&params, &filterParams);
	if (filter == NULL) {
		return -1;
	}

	int status = BNNSFilterApply(filter, x, out);
	BNNSFilterDestroy(filter);
	return status;
}
*/
import "C"

import "unsafe"

// BNNSLinearFP16 performs a matrix-vector multiply with fp16 weights.
// weights: [outDim * inDim] uint16 (fp16, row-major)
// x: [inDim] float32
// out: [outDim] float32
//
// Uses Apple's BNNS framework which can read fp16 weights directly,
// avoiding the fp16->fp32 conversion overhead.
// Returns false if BNNS fails (caller should fall back to fp32 path).
func BNNSLinearFP16(out []float32, weights []uint16, x []float32, outDim, inDim int) bool {
	if outDim <= 0 || inDim <= 0 {
		return false
	}
	if len(out) < outDim || len(weights) < outDim*inDim || len(x) < inDim {
		return false
	}
	status := C.bnns_gemv_f16(
		(*C.float)(unsafe.Pointer(&out[0])),
		(*C.uint16_t)(unsafe.Pointer(&weights[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		C.int(outDim),
		C.int(inDim),
	)
	return status == 0
}
