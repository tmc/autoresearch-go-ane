//go:build darwin

package ane

import (
	xane "github.com/tmc/apple/x/ane"
)

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
			dst[c*width+i] = xane.FP16ToFloat32(src[i])
		}
	}
}

func writeMatrixRowsOffsetFP16Fast(data []uint16, layout xane.TensorLayout, channelOffset, widthOffset, rows, cols int, src []float32) bool {
	return false
}

func writeTransposedMatrixOffsetFP16Fast(data []uint16, layout xane.TensorLayout, channelOffset, widthOffset, srcRows, srcCols int, src []float32) bool {
	return false
}
