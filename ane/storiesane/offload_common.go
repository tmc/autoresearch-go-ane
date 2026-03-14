package storiesane

type tileRange struct {
	start int
	size  int
}

func classifierTileRanges(vocab, tile int) []tileRange {
	if vocab <= 0 || tile <= 0 {
		return nil
	}
	n := (vocab + tile - 1) / tile
	ranges := make([]tileRange, 0, n)
	for start := 0; start < vocab; start += tile {
		size := tile
		if rem := vocab - start; rem < size {
			size = rem
		}
		ranges = append(ranges, tileRange{start: start, size: size})
	}
	return ranges
}
