package stories

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// readBinaryLE reads a binary value from r in little-endian byte order.
func readBinaryLE(r io.Reader, v any) error {
	return binary.Read(r, binary.LittleEndian, v)
}

// LoadMoEPretrained loads a .moe.bin model file.
func LoadMoEPretrained(path string) (*MoEModelWeights, MoEConfig, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, MoEConfig{}, fmt.Errorf("open moe model: %w", err)
	}
	defer f.Close()

	// Read magic
	magic := make([]byte, 4)
	if _, err := io.ReadFull(f, magic); err != nil {
		return nil, MoEConfig{}, fmt.Errorf("read moe magic: %w", err)
	}
	if string(magic) != "MOE1" {
		return nil, MoEConfig{}, fmt.Errorf("bad moe magic: %q", magic)
	}

	// Read JSON config length
	var cfgLen uint32
	if err := readBinaryLE(f, &cfgLen); err != nil {
		return nil, MoEConfig{}, fmt.Errorf("read moe config length: %w", err)
	}
	cfgBytes := make([]byte, cfgLen)
	if _, err := io.ReadFull(f, cfgBytes); err != nil {
		return nil, MoEConfig{}, fmt.Errorf("read moe config: %w", err)
	}

	// Parse JSON config
	var jsonCfg struct {
		Dim              int      `json:"dim"`
		Hidden           int      `json:"hidden"`
		Heads            int      `json:"heads"`
		KVHeads          int      `json:"kv_heads"`
		NLayers          int      `json:"n_layers"`
		Vocab            int      `json:"vocab"`
		Seq              int      `json:"seq"`
		NumExperts       int      `json:"num_experts"`
		NumActiveExperts int      `json:"num_active_experts"`
		ExpertHidden     int      `json:"expert_hidden"`
		HasSharedExpert  bool     `json:"has_shared_expert"`
		LayerTypes       []string `json:"layer_types"`
	}
	if err := json.Unmarshal(cfgBytes, &jsonCfg); err != nil {
		return nil, MoEConfig{}, fmt.Errorf("parse moe config: %w", err)
	}

	cfg := MoEConfig{
		ModelConfig: ModelConfig{
			Dim:     jsonCfg.Dim,
			Hidden:  jsonCfg.Hidden,
			Heads:   jsonCfg.Heads,
			KVHeads: jsonCfg.KVHeads,
			NLayers: jsonCfg.NLayers,
			Vocab:   jsonCfg.Vocab,
			Seq:     jsonCfg.Seq,
		},
		NumExperts:       jsonCfg.NumExperts,
		NumActiveExperts: jsonCfg.NumActiveExperts,
		HasSharedExpert:  jsonCfg.HasSharedExpert,
		ExpertHidden:     jsonCfg.ExpertHidden,
		LayerTypes:       jsonCfg.LayerTypes,
	}

	mw := NewMoEModelWeights(cfg)

	// Read embedding
	if err := readF32s(f, mw.Embed); err != nil {
		return nil, cfg, fmt.Errorf("read moe embed: %w", err)
	}

	// Read per-layer weights
	for i := range mw.Layers {
		layer := &mw.Layers[i]
		if err := readF32s(f, layer.RMSAtt); err != nil {
			return nil, cfg, fmt.Errorf("read moe layer %d rms_att: %w", i, err)
		}
		if err := readF32s(f, layer.Wq); err != nil {
			return nil, cfg, fmt.Errorf("read moe layer %d wq: %w", i, err)
		}
		if err := readF32s(f, layer.Wk); err != nil {
			return nil, cfg, fmt.Errorf("read moe layer %d wk: %w", i, err)
		}
		if err := readF32s(f, layer.Wv); err != nil {
			return nil, cfg, fmt.Errorf("read moe layer %d wv: %w", i, err)
		}
		if err := readF32s(f, layer.Wo); err != nil {
			return nil, cfg, fmt.Errorf("read moe layer %d wo: %w", i, err)
		}
		if err := readF32s(f, layer.RMSFFN); err != nil {
			return nil, cfg, fmt.Errorf("read moe layer %d rms_ffn: %w", i, err)
		}
		if err := readF32s(f, layer.RouterWeight); err != nil {
			return nil, cfg, fmt.Errorf("read moe layer %d router: %w", i, err)
		}
		if cfg.HasSharedExpert && layer.SharedExpert != nil {
			if err := readF32s(f, layer.SharedExpert.W1); err != nil {
				return nil, cfg, fmt.Errorf("read moe layer %d shared w1: %w", i, err)
			}
			if err := readF32s(f, layer.SharedExpert.W2); err != nil {
				return nil, cfg, fmt.Errorf("read moe layer %d shared w2: %w", i, err)
			}
			if err := readF32s(f, layer.SharedExpert.W3); err != nil {
				return nil, cfg, fmt.Errorf("read moe layer %d shared w3: %w", i, err)
			}
		}
		for e := range layer.Experts {
			if err := readF32s(f, layer.Experts[e].W1); err != nil {
				return nil, cfg, fmt.Errorf("read moe layer %d expert %d w1: %w", i, e, err)
			}
			if err := readF32s(f, layer.Experts[e].W2); err != nil {
				return nil, cfg, fmt.Errorf("read moe layer %d expert %d w2: %w", i, e, err)
			}
			if err := readF32s(f, layer.Experts[e].W3); err != nil {
				return nil, cfg, fmt.Errorf("read moe layer %d expert %d w3: %w", i, e, err)
			}
		}
	}

	// Read RMSFinal
	if err := readF32s(f, mw.RMSFinal); err != nil {
		return nil, cfg, fmt.Errorf("read moe rms_final: %w", err)
	}

	return mw, cfg, nil
}
