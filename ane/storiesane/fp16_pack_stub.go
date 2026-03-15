//go:build !darwin || !arm64 || !cgo

package storiesane

import (
	"github.com/tmc/autoresearch-go-ane/ane/mil"
	xane "github.com/tmc/apple/x/ane"
)

// convertFP16ToF32 converts a flat fp16 (uint16) slice to float32 (portable fallback).
func convertFP16ToF32(dst []float32, src []uint16) {
	n := len(src)
	if n > len(dst) {
		n = len(dst)
	}
	for i := 0; i < n; i++ {
		dst[i] = mil.FP16ToFloat32(src[i])
	}
}

// convertF32ToFP16 converts a flat float32 slice to fp16 (uint16) (portable fallback).
func convertF32ToFP16(dst []uint16, src []float32) {
	n := len(src)
	if n > len(dst) {
		n = len(dst)
	}
	for i := 0; i < n; i++ {
		dst[i] = mil.Float32ToFP16(src[i])
	}
}

func writeChannelFirstActsOffsetFP16(data []uint16, layout xane.TensorLayout, channelOffset, widthOffset, width int, x []float32) {
	channels := len(x) / width
	for c := 0; c < channels; c++ {
		row := inputRowFP16(data, layout, channelOffset+c)
		writeContiguousFP16(row[widthOffset:widthOffset+width], x[c*width:(c+1)*width])
	}
}

func readChannelFirstActsOffsetFP16(dst []float32, data []uint16, layout xane.TensorLayout, channelOffset, widthOffset, width int) {
	channels := len(dst) / width
	for c := 0; c < channels; c++ {
		row := inputRowFP16(data, layout, channelOffset+c)
		src := row[widthOffset : widthOffset+width]
		for i := 0; i < width; i++ {
			dst[c*width+i] = mil.FP16ToFloat32(src[i])
		}
	}
}

func writeMatrixRowsOffsetFP16Fast(data []uint16, layout xane.TensorLayout, channelOffset, widthOffset, rows, cols int, src []float32) bool {
	return false
}

func writeTransposedMatrixOffsetFP16Fast(data []uint16, layout xane.TensorLayout, channelOffset, widthOffset, srcRows, srcCols int, src []float32) bool {
	return false
}
