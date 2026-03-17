// Package coordinator provides hardware detection and stable experiment
// identifiers for autoresearch agents.
package coordinator

import (
	"crypto/sha256"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
)

// ChipInfo holds Apple Silicon hardware detection results.
type ChipInfo struct {
	Name string // e.g. "Apple M4 Max"
	Tier string // base, mid, high, ultra
	TOPS int    // ANE compute capability
}

// DetectChip reads the CPU brand string via sysctl and classifies
// the ANE compute tier.
func DetectChip() ChipInfo {
	out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
	if err != nil {
		return ChipInfo{}
	}
	chip := strings.TrimSpace(string(out))
	var tops int
	switch {
	case strings.Contains(chip, "M5"):
		tops = 42
	case strings.Contains(chip, "M4"):
		tops = 38
	case strings.Contains(chip, "M3"):
		tops = 18
	case strings.Contains(chip, "M2"):
		tops = 16
	case strings.Contains(chip, "M1"):
		tops = 11
	}
	tier := "unknown"
	if tops > 0 {
		tier = classifyTier(tops)
	}
	return ChipInfo{Name: chip, Tier: tier, TOPS: tops}
}

// classifyTier maps ANE TOPS to a tier name.
func classifyTier(tops int) string {
	switch {
	case tops <= 12:
		return "base"
	case tops <= 17:
		return "mid"
	case tops <= 20:
		return "high"
	default:
		return "ultra"
	}
}

var reNonAlnum = regexp.MustCompile(`[^a-z0-9]+`)

// Slugify converts text to a URL-safe slug, truncated to maxLen.
func Slugify(text string, maxLen int) string {
	s := reNonAlnum.ReplaceAllString(strings.ToLower(strings.TrimSpace(text)), "-")
	s = strings.Trim(s, "-")
	if len(s) > maxLen {
		s = strings.TrimRight(s[:maxLen], "-")
	}
	return s
}

// ExperimentKey generates a human-readable key: <agent>--<slug>--<hash>.
func ExperimentKey(agent, desc string) string {
	a := Slugify(agent, 20)
	if a == "" {
		a = "unknown"
	}
	h := sha256.Sum256([]byte(strings.ToLower(strings.TrimSpace(desc))))
	return a + "--" + Slugify(desc, 40) + "--" + fmt.Sprintf("%x", h)[:6]
}

// ExperimentHash returns a 12-char hex hash of a description.
func ExperimentHash(desc string) string {
	h := sha256.Sum256([]byte(strings.ToLower(strings.TrimSpace(desc))))
	return fmt.Sprintf("%x", h)[:12]
}
