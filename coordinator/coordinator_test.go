package coordinator

import (
	"strings"
	"testing"
)

func TestSlugify(t *testing.T) {
	tests := []struct {
		name   string
		text   string
		maxLen int
		want   string
	}{
		{"normal", "Hello World", 40, "hello-world"},
		{"special chars", "foo@bar#baz!", 40, "foo-bar-baz"},
		{"long string", "this-is-a-very-long-slug-that-exceeds", 10, "this-is-a"},
		{"empty", "", 40, ""},
		{"only special", "!!!@@@", 40, ""},
		{"leading trailing spaces", "  hello  ", 40, "hello"},
		{"truncate no trailing dash", "abc-def-ghi", 7, "abc-def"},
		{"truncate strips trailing dash", "abc-def-ghi", 4, "abc"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Slugify(tt.text, tt.maxLen); got != tt.want {
				t.Errorf("Slugify(%q, %d) = %q, want %q", tt.text, tt.maxLen, got, tt.want)
			}
		})
	}
}

func TestExperimentKey(t *testing.T) {
	tests := []struct {
		name      string
		agent     string
		desc      string
		wantParts int
	}{
		{"normal", "agent-1", "test ANE prefill", 3},
		{"empty agent", "", "test ANE prefill", 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExperimentKey(tt.agent, tt.desc)
			parts := strings.SplitN(got, "--", 3)
			if len(parts) != tt.wantParts {
				t.Fatalf("ExperimentKey(%q, %q) = %q, want %d parts, got %d", tt.agent, tt.desc, got, tt.wantParts, len(parts))
			}
			if tt.agent == "" && parts[0] != "unknown" {
				t.Errorf("expected 'unknown' agent slug, got %q", parts[0])
			}
			if len(parts[2]) != 6 {
				t.Errorf("hash part length = %d, want 6", len(parts[2]))
			}
		})
	}
}

func TestExperimentHash(t *testing.T) {
	tests := []struct {
		name string
		a, b string
		same bool
	}{
		{"deterministic", "hello world", "hello world", true},
		{"case insensitive", "Hello World", "hello world", true},
		{"whitespace trimmed", "  hello world  ", "hello world", true},
		{"different strings", "alpha", "beta", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ha, hb := ExperimentHash(tt.a), ExperimentHash(tt.b)
			if (ha == hb) != tt.same {
				t.Errorf("ExperimentHash(%q)=%s, ExperimentHash(%q)=%s, same=%v want %v", tt.a, ha, tt.b, hb, ha == hb, tt.same)
			}
			if len(ha) != 12 {
				t.Errorf("hash length = %d, want 12", len(ha))
			}
		})
	}
}

func TestClassifyTier(t *testing.T) {
	tests := []struct {
		name string
		tops int
		want string
	}{
		{"M1 11 TOPS", 11, "base"},
		{"M1 boundary", 12, "base"},
		{"M2 16 TOPS", 16, "mid"},
		{"M2 boundary", 17, "mid"},
		{"M3 18 TOPS", 18, "high"},
		{"M3 boundary", 20, "high"},
		{"M4 38 TOPS", 38, "ultra"},
		{"M5 42 TOPS", 42, "ultra"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := classifyTier(tt.tops); got != tt.want {
				t.Errorf("classifyTier(%d) = %q, want %q", tt.tops, got, tt.want)
			}
		})
	}
}
