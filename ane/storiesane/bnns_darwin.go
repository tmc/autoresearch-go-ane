//go:build darwin && cgo

package storiesane

/*
#cgo darwin CFLAGS: -Wno-deprecated-declarations
#cgo darwin LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
#include <string.h>

// Create a cached BNNS fully-connected filter with fp16 weights.
// Returns NULL on failure. Caller must destroy with BNNSFilterDestroy.
static BNNSFilter bnns_fc_create_f16(
	const uint16_t *weights_f16,
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

	return BNNSFilterCreateLayerFullyConnected(&params, &filterParams);
}

// Apply a cached BNNS filter. Returns 0 on success.
static int bnns_fc_apply(BNNSFilter filter, const float *x, float *out) {
	return BNNSFilterApply(filter, x, out);
}

// Destroy a BNNS filter.
static void bnns_fc_destroy(BNNSFilter filter) {
	if (filter != NULL) {
		BNNSFilterDestroy(filter);
	}
}
*/
import "C"

import "unsafe"

// BNNSFilter wraps a cached Apple BNNS fully-connected filter.
// Create once with fp16 weights, then Apply repeatedly for fast inference.
type BNNSFilter struct {
	filter C.BNNSFilter
}

// NewBNNSFilter creates a cached BNNS fully-connected filter with fp16 weights.
// weights: [outDim * inDim] uint16 (fp16, row-major)
// The filter reads fp16 weights and produces fp32 output.
// Returns nil if creation fails.
func NewBNNSFilter(weights []uint16, outDim, inDim int) *BNNSFilter {
	if outDim <= 0 || inDim <= 0 || len(weights) < outDim*inDim {
		return nil
	}
	f := C.bnns_fc_create_f16(
		(*C.uint16_t)(unsafe.Pointer(&weights[0])),
		C.int(outDim),
		C.int(inDim),
	)
	if f == nil {
		return nil
	}
	return &BNNSFilter{filter: f}
}

// Apply runs the filter: out = weights @ x.
// x: [inDim] float32, out: [outDim] float32.
func (f *BNNSFilter) Apply(out, x []float32) bool {
	if f == nil || f.filter == nil {
		return false
	}
	return C.bnns_fc_apply(f.filter, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&out[0]))) == 0
}

// Close destroys the BNNS filter.
func (f *BNNSFilter) Close() {
	if f != nil && f.filter != nil {
		C.bnns_fc_destroy(f.filter)
		f.filter = nil
	}
}

// BNNSLinearFP16 performs a one-shot matrix-vector multiply with fp16 weights.
// Creates and destroys a filter per call — use NewBNNSFilter for repeated use.
func BNNSLinearFP16(out []float32, weights []uint16, x []float32, outDim, inDim int) bool {
	f := NewBNNSFilter(weights, outDim, inDim)
	if f == nil {
		return false
	}
	defer f.Close()
	return f.Apply(out, x)
}
