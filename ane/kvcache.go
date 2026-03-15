package ane

// kvCache holds cached K and V tensors for all layers, enabling O(n) per-token
// generation instead of O(n^2) full-sequence recomputation.
//
// Layout: each layer's K and V are stored in channel-first order
// [kvDim, maxSeq], where only columns [:pos] are valid. This matches
// the CF layout used by linearCF and the attention kernels.
type kvCache struct {
	// Per-layer K and V caches, each sized [kvDim * maxSeq].
	k [][]float32 // len == nLayers
	v [][]float32 // len == nLayers

	pos    int // number of cached positions (next write index)
	maxSeq int // maximum sequence length
	kvDim  int // KV projection dimension per layer (kvHeads * headDim)
}

func newKVCache(nLayers, kvDim, maxSeq int) *kvCache {
	c := &kvCache{
		k:      make([][]float32, nLayers),
		v:      make([][]float32, nLayers),
		maxSeq: maxSeq,
		kvDim:  kvDim,
	}
	for i := 0; i < nLayers; i++ {
		c.k[i] = make([]float32, kvDim*maxSeq)
		c.v[i] = make([]float32, kvDim*maxSeq)
	}
	return c
}

// appendKV writes a single position's K and V vectors into the cache for the
// given layer, then advances the position counter if this is the last layer.
// newK and newV are in CF layout [kvDim, 1] (i.e., kvDim elements).
//
// The caller must call appendKV for all layers at the same position before
// advancing. Internally we only bump pos after the last layer calls advancePos.
func (c *kvCache) appendKV(layer int, newK, newV []float32) {
	pos := c.pos
	for d := 0; d < c.kvDim; d++ {
		c.k[layer][d*c.maxSeq+pos] = newK[d]
		c.v[layer][d*c.maxSeq+pos] = newV[d]
	}
}

// advancePos increments the cached position counter by one.
func (c *kvCache) advancePos() {
	c.pos++
}

// getK returns the cached K for a layer: [kvDim, pos] in CF layout.
// The returned slice shares memory with the cache; the caller must not modify it.
func (c *kvCache) getK(layer int) []float32 {
	return c.k[layer]
}

// getV returns the cached V for a layer: [kvDim, pos] in CF layout.
func (c *kvCache) getV(layer int) []float32 {
	return c.v[layer]
}

// reset clears the cache so the next call processes from scratch.
func (c *kvCache) reset() {
	c.pos = 0
}
