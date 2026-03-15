package mil

import (
	"fmt"
	"math"
	"strings"
)

// GenGQASDPAForwardInference generates a GQA-aware inference-only attention kernel.
//
// When kvHeads == heads, this produces standard MHA (identical to GenStoriesSDPAForwardInference).
// When kvHeads < heads, K/V projections are smaller and KV heads are repeated to match Q heads.
//
// Input layout per channel row (dim channels):
//
//	x[seq], rms_att[1], Wq[dim], Wk[kvDim], Wv[kvDim], Wo[dim]
//
// Output: x + scale*(Wo@attn - x) (dim*seq channels).
func GenGQASDPAForwardInference(dim, heads, kvHeads, seq int, residualScale float64) string {
	if kvHeads <= 0 || kvHeads == heads {
		// Fall back to standard MHA path.
		return GenStoriesSDPAForwardInference(dim, heads, seq, residualScale)
	}

	headDim := dim / heads
	kvDim := kvHeads * headDim
	scale := 1.0 / math.Sqrt(float64(headDim))
	kvRepeat := heads / kvHeads // How many times to repeat each KV head group
	spatial := seq + 1 + 2*dim + 2*kvDim

	var b strings.Builder
	fmt.Fprintf(&b, "%s{\n", buildInfoHeader)
	fmt.Fprintf(&b, "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> inp) {\n", dim, spatial)
	b.WriteString("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n")
	b.WriteString("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n")
	b.WriteString("        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n")

	// Slice x
	b.WriteString("        tensor<int32, [4]> x_begin = const()[name=string(\"x_begin\"), val=tensor<int32, [4]>([0,0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [4]> x_size = const()[name=string(\"x_size\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> x = slice_by_size(x=inp,begin=x_begin,size=x_size)[name=string(\"x\")];\n", dim, seq)

	// RMSNorm weights
	off := seq
	fmt.Fprintf(&b, "        tensor<int32, [4]> rw_begin = const()[name=string(\"rw_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", off)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rw_size = const()[name=string(\"rw_size\"), val=tensor<int32, [4]>([1,%d,1,1])];\n", dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,1]> rw = slice_by_size(x=inp,begin=rw_begin,size=rw_size)[name=string(\"rw\")];\n", dim)
	off += 1

	// Wq: [dim, dim]
	fmt.Fprintf(&b, "        tensor<int32, [4]> wq_begin = const()[name=string(\"wq_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", off)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wq_size = const()[name=string(\"wq_size\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wq = slice_by_size(x=inp,begin=wq_begin,size=wq_size)[name=string(\"Wq\")];\n", dim, dim)
	off += dim

	// Wk: [dim, kvDim] — smaller for GQA
	fmt.Fprintf(&b, "        tensor<int32, [4]> wk_begin = const()[name=string(\"wk_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", off)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wkv_size = const()[name=string(\"wkv_size\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, kvDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wk = slice_by_size(x=inp,begin=wk_begin,size=wkv_size)[name=string(\"Wk\")];\n", dim, kvDim)
	off += kvDim

	// Wv: [dim, kvDim] — smaller for GQA
	fmt.Fprintf(&b, "        tensor<int32, [4]> wv_begin = const()[name=string(\"wv_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", off)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wv = slice_by_size(x=inp,begin=wv_begin,size=wkv_size)[name=string(\"Wv\")];\n", dim, kvDim)
	off += kvDim

	// Wo: [dim, dim]
	fmt.Fprintf(&b, "        tensor<int32, [4]> wo_begin = const()[name=string(\"wo_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", off)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wo = slice_by_size(x=inp,begin=wo_begin,size=wq_size)[name=string(\"Wo\")];\n", dim, dim)

	// RMSNorm
	appendDynamicRMSNormFP16(&b, "xn", "x", "rw", dim, seq)

	// Q projection: dim -> dim
	appendDynamicMatmulFP16(&b, "qf", "xn", "Wq", dim, dim, seq)
	// K projection: dim -> kvDim
	appendDynamicMatmulFP16(&b, "kf_small", "xn", "Wk", dim, kvDim, seq)
	// V projection: dim -> kvDim
	appendDynamicMatmulFP16(&b, "vf_small", "xn", "Wv", dim, kvDim, seq)

	// Reshape Q to multi-head: [1, heads, headDim, seq] -> [1, heads, seq, headDim]
	fmt.Fprintf(&b, "        tensor<int32, [4]> q_shape = const()[name=string(\"q_shape\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=q_shape,x=qf)[name=string(\"q4\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=string(\"q\")];\n", heads, seq, headDim)

	// Reshape K to KV-head: [1, kvHeads, headDim, seq] -> [1, kvHeads, seq, headDim]
	fmt.Fprintf(&b, "        tensor<int32, [4]> kv_shape = const()[name=string(\"kv_shape\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", kvHeads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=kv_shape,x=kf_small)[name=string(\"k4\")];\n", kvHeads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_kv = transpose(perm=pm,x=k4)[name=string(\"k_kv\")];\n", kvHeads, seq, headDim)

	// Reshape V to KV-head
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=kv_shape,x=vf_small)[name=string(\"v4\")];\n", kvHeads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v_kv = transpose(perm=pm,x=v4)[name=string(\"v_kv\")];\n", kvHeads, seq, headDim)

	// RoPE tables
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> rope_cos = const()[name=string(\"rope_cos\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];\n", seq, headDim, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> rope_sin = const()[name=string(\"rope_sin\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];\n", seq, headDim, seq, headDim)

	// RoPE constants
	qPairs := seq * headDim / 2
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_sh_q = const()[name=string(\"rp_sh_q\"), val=tensor<int32, [4]>([1,%d,%d,2])];\n", heads, qPairs)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_s1_q = const()[name=string(\"rp_s1_q\"), val=tensor<int32, [4]>([1,%d,%d,1])];\n", heads, qPairs)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_bk_q = const()[name=string(\"rp_bk_q\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", heads, seq, headDim)
	kvPairs := seq * headDim / 2
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_sh_kv = const()[name=string(\"rp_sh_kv\"), val=tensor<int32, [4]>([1,%d,%d,2])];\n", kvHeads, kvPairs)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_s1_kv = const()[name=string(\"rp_s1_kv\"), val=tensor<int32, [4]>([1,%d,%d,1])];\n", kvHeads, kvPairs)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_bk_kv = const()[name=string(\"rp_bk_kv\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", kvHeads, seq, headDim)
	b.WriteString("        tensor<int32, [4]> rp_b0 = const()[name=string(\"rp_b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n")
	b.WriteString("        tensor<int32, [4]> rp_b1 = const()[name=string(\"rp_b1\"), val=tensor<int32, [4]>([0,0,0,1])];\n")
	b.WriteString("        fp16 neg1 = const()[name=string(\"neg1\"), val=fp16(-1.0)];\n")
	b.WriteString("        int32 rpax = const()[name=string(\"rpax\"), val=int32(3)];\n")
	b.WriteString("        bool rpil = const()[name=string(\"rpil\"), val=bool(false)];\n")

	// Apply RoPE to Q (heads heads)
	appendRoPEInPlace(&b, "q", "q", heads, qPairs, seq, headDim, "rp_sh_q", "rp_s1_q", "rp_bk_q")

	// Apply RoPE to K (kvHeads heads)
	appendRoPEInPlace(&b, "k", "k_kv", kvHeads, kvPairs, seq, headDim, "rp_sh_kv", "rp_s1_kv", "rp_bk_kv")

	// Expand KV heads to match Q heads by tiling along batch/head dim.
	// tile op repeats along specified axes: [1, kvHeads, seq, headDim] -> [1, heads, seq, headDim]
	fmt.Fprintf(&b, "        tensor<int32, [4]> kv_reps = const()[name=string(\"kv_reps\"), val=tensor<int32, [4]>([1,%d,1,1])];\n", kvRepeat)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k = tile(reps=kv_reps,x=k_rope)[name=string(\"k\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v = tile(reps=kv_reps,x=v_kv)[name=string(\"v\")];\n", heads, seq, headDim)

	// Scaled dot-product attention (now all heads match)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q_rope,y=k)[name=string(\"sc1\")];\n", heads, seq, seq)
	fmt.Fprintf(&b, "        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", scale)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];\n", heads, seq, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", seq, seq, seq, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];\n", heads, seq, seq)
	b.WriteString("        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms)[name=string(\"aw\")];\n", heads, seq, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=v)[name=string(\"a4\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> at = transpose(perm=pm,x=a4)[name=string(\"at\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> att_shape = const()[name=string(\"att_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> af = reshape(shape=att_shape,x=at)[name=string(\"af\")];\n", dim, seq)

	// Wo projection
	appendDynamicMatmulFP16(&b, "oo", "af", "Wo", dim, dim, seq)

	// Residual with scale: out = x + scale*(oo - x)
	fmt.Fprintf(&b, "        fp16 rs = const()[name=string(\"rs\"), val=fp16(%f)];\n", residualScale)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> oo_diff = sub(x=oo,y=x)[name=string(\"oo_diff\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> oo_ds = mul(x=oo_diff,y=rs)[name=string(\"oo_ds\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> out = add(x=x,y=oo_ds)[name=string(\"out\")];\n", dim, seq)
	b.WriteString("    } -> (out);\n}\n")
	return b.String()
}

// appendRoPEInPlace emits MIL ops to apply RoPE to a tensor in-place.
// prefix is the output variable name, inVar is the input variable name.
// The output variable is named prefix+"_rope".
func appendRoPEInPlace(b *strings.Builder, prefix, inVar string, nHeads, pairs, seq, headDim int, shapeVar, s1Var, bkVar string) {
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,%d,2]> %s_pair = reshape(shape=%s,x=%s)[name=string(\"%s_pair\")];\n", nHeads, pairs, prefix, shapeVar, inVar, prefix)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,%d,1]> %s_e = slice_by_size(x=%s_pair,begin=rp_b0,size=%s)[name=string(\"%s_e\")];\n", nHeads, pairs, prefix, prefix, s1Var, prefix)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,%d,1]> %s_o = slice_by_size(x=%s_pair,begin=rp_b1,size=%s)[name=string(\"%s_o\")];\n", nHeads, pairs, prefix, prefix, s1Var, prefix)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,%d,1]> %s_no = mul(x=%s_o,y=neg1)[name=string(\"%s_no\")];\n", nHeads, pairs, prefix, prefix, prefix)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,%d,2]> %s_rotp = concat(axis=rpax,interleave=rpil,values=(%s_no,%s_e))[name=string(\"%s_rotp\")];\n", nHeads, pairs, prefix, prefix, prefix, prefix)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,%d,%d]> %s_rot = reshape(shape=%s,x=%s_rotp)[name=string(\"%s_rot\")];\n", nHeads, seq, headDim, prefix, bkVar, prefix, prefix)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,%d,%d]> %s_c = mul(x=%s,y=rope_cos)[name=string(\"%s_c\")];\n", nHeads, seq, headDim, prefix, inVar, prefix)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,%d,%d]> %s_rs = mul(x=%s_rot,y=rope_sin)[name=string(\"%s_rs\")];\n", nHeads, seq, headDim, prefix, prefix, prefix)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,%d,%d]> %s_rope = add(x=%s_c,y=%s_rs)[name=string(\"%s_rope\")];\n", nHeads, seq, headDim, prefix, prefix, prefix, prefix)
}
