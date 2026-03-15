package mil

import (
	"fmt"
	"math"
	"strings"
)

func appendDynamicMatmulFP16(b *strings.Builder, prefix, actVar, wtVar string, inCh, outCh, seq int) {
	fmt.Fprintf(b, "        tensor<int32, [4]> %s_act_shape = const()[name=string(\"%s_act_shape\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", prefix, prefix, inCh, seq)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,%d,%d]> %s_act2 = reshape(shape=%s_act_shape,x=%s)[name=string(\"%s_act2\")];\n", inCh, seq, prefix, prefix, actVar, prefix)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,%d,%d]> %s_act3 = transpose(perm=pm,x=%s_act2)[name=string(\"%s_act3\")];\n", seq, inCh, prefix, prefix, prefix)
	fmt.Fprintf(b, "        tensor<int32, [4]> %s_wt_shape = const()[name=string(\"%s_wt_shape\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", prefix, prefix, inCh, outCh)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,%d,%d]> %s_wt2 = reshape(shape=%s_wt_shape,x=%s)[name=string(\"%s_wt2\")];\n", inCh, outCh, prefix, prefix, wtVar, prefix)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,%d,%d]> %s_mm = matmul(transpose_x=bF,transpose_y=bF,x=%s_act3,y=%s_wt2)[name=string(\"%s_mm\")];\n", seq, outCh, prefix, prefix, prefix, prefix)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,%d,%d]> %s_mm_t = transpose(perm=pm,x=%s_mm)[name=string(\"%s_mm_t\")];\n", outCh, seq, prefix, prefix, prefix)
	fmt.Fprintf(b, "        tensor<int32, [4]> %s_out_shape = const()[name=string(\"%s_out_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", prefix, prefix, outCh, seq)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,%d]> %s = reshape(shape=%s_out_shape,x=%s_mm_t)[name=string(\"%s\")];\n", outCh, seq, prefix, prefix, prefix, prefix)
}

func appendDynamicRMSNormFP16(b *strings.Builder, outVar, xVar, wVar string, dim, seq int) {
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,%d]> %s_sq = mul(x=%s,y=%s)[name=string(\"%s_sq\")];\n", dim, seq, outVar, xVar, xVar, outVar)
	b.WriteString("        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n")
	b.WriteString("        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n")
	fmt.Fprintf(b, "        tensor<fp16, [1,1,1,%d]> %s_ss2 = reduce_mean(x=%s_sq,axes=rax,keep_dims=kd)[name=string(\"%s_ss2\")];\n", seq, outVar, outVar, outVar)
	b.WriteString("        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n")
	fmt.Fprintf(b, "        tensor<fp16, [1,1,1,%d]> %s_ss3 = add(x=%s_ss2,y=eps)[name=string(\"%s_ss3\")];\n", seq, outVar, outVar, outVar)
	fmt.Fprintf(b, "        tensor<fp16, [1,1,1,%d]> %s_rrms = rsqrt(x=%s_ss3)[name=string(\"%s_rrms\")];\n", seq, outVar, outVar, outVar)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,%d]> %s_xr = mul(x=%s,y=%s_rrms)[name=string(\"%s_xr\")];\n", dim, seq, outVar, xVar, outVar, outVar)
	fmt.Fprintf(b, "        tensor<fp16, [1,%d,1,%d]> %s = mul(x=%s_xr,y=%s)[name=string(\"%s\")];\n", dim, seq, outVar, outVar, wVar, outVar)
}

// GenDynamicMatmulFP16 generates a single-input fp16 matmul kernel.
//
// The input tensor is [1, inCh, 1, seq+outCh] fp16:
//   - spatial [0:seq] contains activations in channel-first layout [inCh, seq]
//   - spatial [seq:seq+outCh] contains weights in row-major [inCh, outCh]
//
// The output tensor is [1, outCh, 1, seq] fp16.
func GenDynamicMatmulFP16(inCh, outCh, seq int) string {
	spatial := seq + outCh
	var b strings.Builder
	fmt.Fprintf(&b, "%s{\n", buildInfoHeader)
	fmt.Fprintf(&b, "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", inCh, spatial)
	b.WriteString("        tensor<int32, [4]> act_begin = const()[name=string(\"act_begin\"), val=tensor<int32, [4]>([0,0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [4]> act_size = const()[name=string(\"act_size\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", inCh, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=act_begin,size=act_size)[name=string(\"act\")];\n", inCh, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wt_begin = const()[name=string(\"wt_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wt_size = const()[name=string(\"wt_size\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", inCh, outCh)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=wt_begin,size=wt_size)[name=string(\"wt\")];\n", inCh, outCh)
	b.WriteString("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n")
	b.WriteString("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n")
	appendDynamicMatmulFP16(&b, "out", "act", "wt", inCh, outCh, seq)
	b.WriteString("    } -> (out);\n}\n")
	return b.String()
}

// GenStoriesSDPAForwardDynamicTaps generates a packed attention-forward kernel.
//
// Input layout per input channel row:
//
//	x[seq], rms_att[1], Wq[row dim], Wk[row dim], Wv[row dim], Wo[row dim]
//
// Output layout:
//
//	concat(x2, q, k, v, att_out) along channels.
func GenStoriesSDPAForwardDynamicTaps(dim, heads, seq int) string {
	headDim := dim / heads
	scale := 1.0 / math.Sqrt(float64(headDim))
	spatial := seq + 1 + 4*dim

	var b strings.Builder
	fmt.Fprintf(&b, "%s{\n", buildInfoHeader)
	fmt.Fprintf(&b, "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> inp) {\n", dim, spatial)
	b.WriteString("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n")
	b.WriteString("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n")
	b.WriteString("        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n")
	b.WriteString("        tensor<int32, [4]> x_begin = const()[name=string(\"x_begin\"), val=tensor<int32, [4]>([0,0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [4]> x_size = const()[name=string(\"x_size\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> x = slice_by_size(x=inp,begin=x_begin,size=x_size)[name=string(\"x\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rw_begin = const()[name=string(\"rw_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rw_size = const()[name=string(\"rw_size\"), val=tensor<int32, [4]>([1,%d,1,1])];\n", dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,1]> rw = slice_by_size(x=inp,begin=rw_begin,size=rw_size)[name=string(\"rw\")];\n", dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wq_begin = const()[name=string(\"wq_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq+1)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wt_size = const()[name=string(\"wt_size\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wq = slice_by_size(x=inp,begin=wq_begin,size=wt_size)[name=string(\"Wq\")];\n", dim, dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wk_begin = const()[name=string(\"wk_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq+1+dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wk = slice_by_size(x=inp,begin=wk_begin,size=wt_size)[name=string(\"Wk\")];\n", dim, dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wv_begin = const()[name=string(\"wv_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq+1+2*dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wv = slice_by_size(x=inp,begin=wv_begin,size=wt_size)[name=string(\"Wv\")];\n", dim, dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wo_begin = const()[name=string(\"wo_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq+1+3*dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wo = slice_by_size(x=inp,begin=wo_begin,size=wt_size)[name=string(\"Wo\")];\n", dim, dim)
	appendDynamicRMSNormFP16(&b, "xn", "x", "rw", dim, seq)
	appendDynamicMatmulFP16(&b, "qf", "xn", "Wq", dim, dim, seq)
	appendDynamicMatmulFP16(&b, "kf", "xn", "Wk", dim, dim, seq)
	appendDynamicMatmulFP16(&b, "vf", "xn", "Wv", dim, dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> q_shape = const()[name=string(\"q_shape\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=q_shape,x=qf)[name=string(\"q4\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=string(\"q\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=q_shape,x=kf)[name=string(\"k4\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=k4)[name=string(\"k\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=q_shape,x=vf)[name=string(\"v4\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=v4)[name=string(\"v\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> rope_cos = const()[name=string(\"rope_cos\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];\n", seq, headDim, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> rope_sin = const()[name=string(\"rope_sin\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];\n", seq, headDim, seq, headDim)
	pairs := seq * headDim / 2
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_sh = const()[name=string(\"rp_sh\"), val=tensor<int32, [4]>([1,%d,%d,2])];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_s1 = const()[name=string(\"rp_s1\"), val=tensor<int32, [4]>([1,%d,%d,1])];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_bk = const()[name=string(\"rp_bk\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", heads, seq, headDim)
	b.WriteString("        tensor<int32, [4]> rp_b0 = const()[name=string(\"rp_b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n")
	b.WriteString("        tensor<int32, [4]> rp_b1 = const()[name=string(\"rp_b1\"), val=tensor<int32, [4]>([0,0,0,1])];\n")
	b.WriteString("        fp16 neg1 = const()[name=string(\"neg1\"), val=fp16(-1.0)];\n")
	b.WriteString("        int32 rpax = const()[name=string(\"rpax\"), val=int32(3)];\n")
	b.WriteString("        bool rpil = const()[name=string(\"rpil\"), val=bool(false)];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,2]> q_pair = reshape(shape=rp_sh,x=q)[name=string(\"q_pair\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> q_e = slice_by_size(x=q_pair,begin=rp_b0,size=rp_s1)[name=string(\"q_e\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> q_o = slice_by_size(x=q_pair,begin=rp_b1,size=rp_s1)[name=string(\"q_o\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> q_no = mul(x=q_o,y=neg1)[name=string(\"q_no\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,2]> q_rotp = concat(axis=rpax,interleave=rpil,values=(q_no,q_e))[name=string(\"q_rotp\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q_rot = reshape(shape=rp_bk,x=q_rotp)[name=string(\"q_rot\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q_c = mul(x=q,y=rope_cos)[name=string(\"q_c\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q_rs = mul(x=q_rot,y=rope_sin)[name=string(\"q_rs\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q_rope = add(x=q_c,y=q_rs)[name=string(\"q_rope\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,2]> k_pair = reshape(shape=rp_sh,x=k)[name=string(\"k_pair\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> k_e = slice_by_size(x=k_pair,begin=rp_b0,size=rp_s1)[name=string(\"k_e\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> k_o = slice_by_size(x=k_pair,begin=rp_b1,size=rp_s1)[name=string(\"k_o\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> k_no = mul(x=k_o,y=neg1)[name=string(\"k_no\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,2]> k_rotp = concat(axis=rpax,interleave=rpil,values=(k_no,k_e))[name=string(\"k_rotp\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_rot = reshape(shape=rp_bk,x=k_rotp)[name=string(\"k_rot\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_c = mul(x=k,y=rope_cos)[name=string(\"k_c\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_rs = mul(x=k_rot,y=rope_sin)[name=string(\"k_rs\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_rope = add(x=k_c,y=k_rs)[name=string(\"k_rope\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q_rope,y=k_rope)[name=string(\"sc1\")];\n", heads, seq, seq)
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
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q_rt = transpose(perm=pm,x=q_rope)[name=string(\"q_rt\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> qrf = reshape(shape=att_shape,x=q_rt)[name=string(\"qrf\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_rt = transpose(perm=pm,x=k_rope)[name=string(\"k_rt\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> krf = reshape(shape=att_shape,x=k_rt)[name=string(\"krf\")];\n", dim, seq)
	appendDynamicMatmulFP16(&b, "oo", "af", "Wo", dim, dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> x2 = add(x=x,y=oo)[name=string(\"x2\")];\n", dim, seq)
	b.WriteString("        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n")
	b.WriteString("        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(x2,qrf,krf,vf,af))[name=string(\"out\")];\n", 5*dim, seq)
	b.WriteString("    } -> (out);\n}\n")
	return b.String()
}

// GenStoriesFFNForwardDynamicTaps generates a packed FFN-forward kernel.
//
// Input layout per input channel row:
//
//	x2[seq], rms_ffn[1], W1[row hidden], W3[row hidden], W2[row hidden]
//
// Output layout:
//
//	concat(x_next, h1, h3) along channels.
func GenStoriesFFNForwardDynamicTaps(dim, hidden, seq int) string {
	spatial := seq + 1 + 3*hidden

	var b strings.Builder
	fmt.Fprintf(&b, "%s{\n", buildInfoHeader)
	fmt.Fprintf(&b, "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> inp) {\n", dim, spatial)
	b.WriteString("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n")
	b.WriteString("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n")
	b.WriteString("        tensor<int32, [4]> x_begin = const()[name=string(\"x_begin\"), val=tensor<int32, [4]>([0,0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [4]> x_size = const()[name=string(\"x_size\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> x2 = slice_by_size(x=inp,begin=x_begin,size=x_size)[name=string(\"x2\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rw_begin = const()[name=string(\"rw_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rw_size = const()[name=string(\"rw_size\"), val=tensor<int32, [4]>([1,%d,1,1])];\n", dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,1]> rw = slice_by_size(x=inp,begin=rw_begin,size=rw_size)[name=string(\"rw\")];\n", dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> w_shape = const()[name=string(\"w_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, hidden)
	fmt.Fprintf(&b, "        tensor<int32, [4]> w1_begin = const()[name=string(\"w1_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq+1)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> W1 = slice_by_size(x=inp,begin=w1_begin,size=w_shape)[name=string(\"W1\")];\n", dim, hidden)
	fmt.Fprintf(&b, "        tensor<int32, [4]> w3_begin = const()[name=string(\"w3_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq+1+hidden)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> W3 = slice_by_size(x=inp,begin=w3_begin,size=w_shape)[name=string(\"W3\")];\n", dim, hidden)
	fmt.Fprintf(&b, "        tensor<int32, [4]> w2_begin = const()[name=string(\"w2_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq+1+2*hidden)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> W2r = slice_by_size(x=inp,begin=w2_begin,size=w_shape)[name=string(\"W2r\")];\n", dim, hidden)
	appendDynamicRMSNormFP16(&b, "xn", "x2", "rw", dim, seq)
	appendDynamicMatmulFP16(&b, "h1", "xn", "W1", dim, hidden, seq)
	appendDynamicMatmulFP16(&b, "h3", "xn", "W3", dim, hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> silu_h1 = silu(x=h1)[name=string(\"silu_h1\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu_h1,y=h3)[name=string(\"gate\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> gate_shape = const()[name=string(\"gate_shape\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> gate2 = reshape(shape=gate_shape,x=gate)[name=string(\"gate2\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> gate3 = transpose(perm=pm,x=gate2)[name=string(\"gate3\")];\n", seq, hidden)
	fmt.Fprintf(&b, "        tensor<int32, [4]> w2_shape = const()[name=string(\"w2_shape\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", dim, hidden)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> W22 = reshape(shape=w2_shape,x=W2r)[name=string(\"W22\")];\n", dim, hidden)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> W2 = transpose(perm=pm,x=W22)[name=string(\"W2\")];\n", hidden, dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> ffm = matmul(transpose_x=bF,transpose_y=bF,x=gate3,y=W2)[name=string(\"ffm\")];\n", seq, dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> fft = transpose(perm=pm,x=ffm)[name=string(\"fft\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> ff_shape = const()[name=string(\"ff_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> ff = reshape(shape=ff_shape,x=fft)[name=string(\"ff\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> x_next = add(x=x2,y=ff)[name=string(\"x_next\")];\n", dim, seq)
	b.WriteString("        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n")
	b.WriteString("        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(x_next,h1,h3))[name=string(\"out\")];\n", dim+2*hidden, seq)
	b.WriteString("    } -> (out);\n}\n")
	return b.String()
}

// GenStoriesFFNBackwardTailDynamic generates the packed FFN backward tail.
//
// Input layout per hidden channel row:
//
//	dsilu[seq], h1[seq], h3[seq], W1[row dim], W3[row dim]
//
// Output layout:
//
//	concat(dx, dh1, dh3) along channels.
func GenStoriesFFNBackwardTailDynamic(dim, hidden, seq int) string {
	spatial := 3*seq + 2*dim

	var b strings.Builder
	fmt.Fprintf(&b, "%s{\n", buildInfoHeader)
	fmt.Fprintf(&b, "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> inp) {\n", hidden, spatial)
	b.WriteString("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n")
	b.WriteString("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n")
	fmt.Fprintf(&b, "        tensor<int32, [4]> sh = const()[name=string(\"sh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", hidden, seq)
	b.WriteString("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> dsilu = slice_by_size(x=inp,begin=b0,size=sh)[name=string(\"dsilu\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> h1 = slice_by_size(x=inp,begin=b1,size=sh)[name=string(\"h1\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> h3 = slice_by_size(x=inp,begin=b2,size=sh)[name=string(\"h3\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wt_shape = const()[name=string(\"wt_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", hidden, dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 3*seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> W1 = slice_by_size(x=inp,begin=b3,size=wt_shape)[name=string(\"W1\")];\n", hidden, dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> b4 = const()[name=string(\"b4\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 3*seq+dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> W3 = slice_by_size(x=inp,begin=b4,size=wt_shape)[name=string(\"W3\")];\n", hidden, dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sig\")];\n", hidden, seq)
	b.WriteString("        fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> oms = sub(x=one,y=sig)[name=string(\"oms\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> homs = mul(x=h1,y=oms)[name=string(\"homs\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> brk = add(x=one,y=homs)[name=string(\"brk\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> dsd = mul(x=sig,y=brk)[name=string(\"dsd\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> t1 = mul(x=dsilu,y=h3)[name=string(\"t1\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> dh1 = mul(x=t1,y=dsd)[name=string(\"dh1\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> slh = mul(x=h1,y=sig)[name=string(\"slh\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> dh3 = mul(x=dsilu,y=slh)[name=string(\"dh3\")];\n", hidden, seq)
	appendDynamicMatmulFP16(&b, "dx1", "dh1", "W1", hidden, dim, seq)
	appendDynamicMatmulFP16(&b, "dx3", "dh3", "W3", hidden, dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> dx = add(x=dx1,y=dx3)[name=string(\"dx\")];\n", dim, seq)
	b.WriteString("        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n")
	b.WriteString("        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dx,dh1,dh3))[name=string(\"out\")];\n", dim+2*hidden, seq)
	b.WriteString("    } -> (out);\n}\n")
	return b.String()
}

// GenStoriesSDPABackward1Dynamic generates the first attention backward stage.
//
// Input layout is concat(q, k, v, da) along channels.
// Output layout is concat(dv, probs, dp) along channels.
func GenStoriesSDPABackward1Dynamic(dim, heads, seq int) string {
	headDim := dim / heads
	scoreCh := heads * seq
	scale := 1.0 / math.Sqrt(float64(headDim))

	var b strings.Builder
	fmt.Fprintf(&b, "%s{\n", buildInfoHeader)
	fmt.Fprintf(&b, "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", 4*dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq)
	b.WriteString("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b0,size=sz)[name=string(\"qf\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b1,size=sz)[name=string(\"kf\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> vf = slice_by_size(x=x,begin=b2,size=sz)[name=string(\"vf\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 3*dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> da = slice_by_size(x=x,begin=b3,size=sz)[name=string(\"da\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", heads, headDim, seq)
	b.WriteString("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=string(\"qr\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=string(\"q\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf)[name=string(\"kr\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr)[name=string(\"k\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> vr = reshape(shape=rsh,x=vf)[name=string(\"vr\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=vr)[name=string(\"v\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> dr = reshape(shape=rsh,x=da)[name=string(\"dr\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> d = transpose(perm=pm,x=dr)[name=string(\"d\")];\n", heads, seq, headDim)
	b.WriteString("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n")
	b.WriteString("        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"sc1\")];\n", heads, seq, seq)
	fmt.Fprintf(&b, "        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", scale)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"sc2\")];\n", heads, seq, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", seq, seq, seq, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"ms\")];\n", heads, seq, seq)
	b.WriteString("        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> probs = softmax(axis=sax,x=ms)[name=string(\"probs\")];\n", heads, seq, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=d)[name=string(\"dv4\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> dp4 = matmul(transpose_x=bF,transpose_y=bT,x=d,y=v)[name=string(\"dp4\")];\n", heads, seq, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> dvt = transpose(perm=pm,x=dv4)[name=string(\"dvt\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> dv_shape = const()[name=string(\"dv_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> dvf = reshape(shape=dv_shape,x=dvt)[name=string(\"dvf\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> sc_shape = const()[name=string(\"sc_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", scoreCh, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> pf = reshape(shape=sc_shape,x=probs)[name=string(\"pf\")];\n", scoreCh, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> dpf = reshape(shape=sc_shape,x=dp4)[name=string(\"dpf\")];\n", scoreCh, seq)
	b.WriteString("        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n")
	b.WriteString("        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dvf,pf,dpf))[name=string(\"out\")];\n", dim+2*scoreCh, seq)
	b.WriteString("    } -> (out);\n}\n")
	return b.String()
}

// GenStoriesQKVBackwardDynamic generates the packed QKV backward dx kernel.
//
// Input layout per channel row:
//
//	dq[seq], dk[seq], dv[seq], Wq[row dim], Wk[row dim], Wv[row dim]
//
// Output layout:
//
//	dx along channels.
func GenStoriesQKVBackwardDynamic(dim, heads, seq int) string {
	_ = heads
	spatial := 3*seq + 3*dim

	var b strings.Builder
	fmt.Fprintf(&b, "%s{\n", buildInfoHeader)
	fmt.Fprintf(&b, "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> inp) {\n", dim, spatial)
	b.WriteString("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n")
	b.WriteString("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n")
	fmt.Fprintf(&b, "        tensor<int32, [4]> grad_shape = const()[name=string(\"grad_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq)
	b.WriteString("        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> dq = slice_by_size(x=inp,begin=b0,size=grad_shape)[name=string(\"dq\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> dk = slice_by_size(x=inp,begin=b1,size=grad_shape)[name=string(\"dk\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> dv = slice_by_size(x=inp,begin=b2,size=grad_shape)[name=string(\"dv\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wt_shape = const()[name=string(\"wt_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wq_begin = const()[name=string(\"wq_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 3*seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wq = slice_by_size(x=inp,begin=wq_begin,size=wt_shape)[name=string(\"Wq\")];\n", dim, dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wk_begin = const()[name=string(\"wk_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 3*seq+dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wk = slice_by_size(x=inp,begin=wk_begin,size=wt_shape)[name=string(\"Wk\")];\n", dim, dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wv_begin = const()[name=string(\"wv_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 3*seq+2*dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wv = slice_by_size(x=inp,begin=wv_begin,size=wt_shape)[name=string(\"Wv\")];\n", dim, dim)
	appendDynamicMatmulFP16(&b, "dxq", "dq", "Wq", dim, dim, seq)
	appendDynamicMatmulFP16(&b, "dxk", "dk", "Wk", dim, dim, seq)
	appendDynamicMatmulFP16(&b, "dxv", "dv", "Wv", dim, dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> dxqk = add(x=dxq,y=dxk)[name=string(\"dxqk\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> out = add(x=dxqk,y=dxv)[name=string(\"out\")];\n", dim, seq)
	b.WriteString("    } -> (out);\n}\n")
	return b.String()
}

// GenStoriesFullLayerInference generates a fused attention+FFN kernel.
// Eliminates the IOSurface round-trip between attention and FFN.
//
// Input layout per channel row (dim channels):
//
//	x[seq], rms_att[1], Wq[dim], Wk[dim], Wv[dim], Wo[dim], rms_ffn[1], W1[hidden], W3[hidden], W2_rows[hidden]
//
// Output: concat(x_next, q, k, v, att_out, h1, h3) for backward taps, or just x_next for inference.
func GenStoriesFullLayerInference(dim, hidden, heads, seq int) string {
	headDim := dim / heads
	scale := 1.0 / math.Sqrt(float64(headDim))
	spatial := seq + 1 + 4*dim + 1 + 3*hidden

	var b strings.Builder
	fmt.Fprintf(&b, "%s{\n", buildInfoHeader)
	fmt.Fprintf(&b, "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> inp) {\n", dim, spatial)
	b.WriteString("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n")
	b.WriteString("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n")
	b.WriteString("        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n")

	// Slice activations
	b.WriteString("        tensor<int32, [4]> x_begin = const()[name=string(\"x_begin\"), val=tensor<int32, [4]>([0,0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [4]> x_size = const()[name=string(\"x_size\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> x = slice_by_size(x=inp,begin=x_begin,size=x_size)[name=string(\"x\")];\n", dim, seq)

	// Attention weights
	off := seq
	fmt.Fprintf(&b, "        tensor<int32, [4]> rw_begin = const()[name=string(\"rw_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", off)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rw_size = const()[name=string(\"rw_size\"), val=tensor<int32, [4]>([1,%d,1,1])];\n", dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,1]> rw = slice_by_size(x=inp,begin=rw_begin,size=rw_size)[name=string(\"rw\")];\n", dim)
	off += 1
	fmt.Fprintf(&b, "        tensor<int32, [4]> wt_size = const()[name=string(\"wt_size\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wq_begin = const()[name=string(\"wq_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", off)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wq = slice_by_size(x=inp,begin=wq_begin,size=wt_size)[name=string(\"Wq\")];\n", dim, dim)
	off += dim
	fmt.Fprintf(&b, "        tensor<int32, [4]> wk_begin = const()[name=string(\"wk_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", off)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wk = slice_by_size(x=inp,begin=wk_begin,size=wt_size)[name=string(\"Wk\")];\n", dim, dim)
	off += dim
	fmt.Fprintf(&b, "        tensor<int32, [4]> wv_begin = const()[name=string(\"wv_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", off)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wv = slice_by_size(x=inp,begin=wv_begin,size=wt_size)[name=string(\"Wv\")];\n", dim, dim)
	off += dim
	fmt.Fprintf(&b, "        tensor<int32, [4]> wo_begin = const()[name=string(\"wo_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", off)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wo = slice_by_size(x=inp,begin=wo_begin,size=wt_size)[name=string(\"Wo\")];\n", dim, dim)
	off += dim

	// FFN weights
	fmt.Fprintf(&b, "        tensor<int32, [4]> rw2_begin = const()[name=string(\"rw2_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", off)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,1]> rw2 = slice_by_size(x=inp,begin=rw2_begin,size=rw_size)[name=string(\"rw2\")];\n", dim)
	off += 1
	fmt.Fprintf(&b, "        tensor<int32, [4]> w_shape = const()[name=string(\"w_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, hidden)
	fmt.Fprintf(&b, "        tensor<int32, [4]> w1_begin = const()[name=string(\"w1_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", off)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> W1 = slice_by_size(x=inp,begin=w1_begin,size=w_shape)[name=string(\"W1\")];\n", dim, hidden)
	off += hidden
	fmt.Fprintf(&b, "        tensor<int32, [4]> w3_begin = const()[name=string(\"w3_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", off)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> W3 = slice_by_size(x=inp,begin=w3_begin,size=w_shape)[name=string(\"W3\")];\n", dim, hidden)
	off += hidden
	fmt.Fprintf(&b, "        tensor<int32, [4]> w2_begin = const()[name=string(\"w2_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", off)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> W2r = slice_by_size(x=inp,begin=w2_begin,size=w_shape)[name=string(\"W2r\")];\n", dim, hidden)

	// === ATTENTION BLOCK ===
	appendDynamicRMSNormFP16(&b, "xn", "x", "rw", dim, seq)
	appendDynamicMatmulFP16(&b, "qf", "xn", "Wq", dim, dim, seq)
	appendDynamicMatmulFP16(&b, "kf", "xn", "Wk", dim, dim, seq)
	appendDynamicMatmulFP16(&b, "vf", "xn", "Wv", dim, dim, seq)

	// Multi-head reshape + RoPE
	fmt.Fprintf(&b, "        tensor<int32, [4]> q_shape = const()[name=string(\"q_shape\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=q_shape,x=qf)[name=string(\"q4\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=string(\"q\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=q_shape,x=kf)[name=string(\"k4\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=k4)[name=string(\"k\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=q_shape,x=vf)[name=string(\"v4\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=v4)[name=string(\"v\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> rope_cos = const()[name=string(\"rope_cos\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];\n", seq, headDim, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> rope_sin = const()[name=string(\"rope_sin\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];\n", seq, headDim, seq, headDim)
	pairs := seq * headDim / 2
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_sh = const()[name=string(\"rp_sh\"), val=tensor<int32, [4]>([1,%d,%d,2])];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_s1 = const()[name=string(\"rp_s1\"), val=tensor<int32, [4]>([1,%d,%d,1])];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_bk = const()[name=string(\"rp_bk\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", heads, seq, headDim)
	b.WriteString("        tensor<int32, [4]> rp_b0 = const()[name=string(\"rp_b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n")
	b.WriteString("        tensor<int32, [4]> rp_b1 = const()[name=string(\"rp_b1\"), val=tensor<int32, [4]>([0,0,0,1])];\n")
	b.WriteString("        fp16 neg1 = const()[name=string(\"neg1\"), val=fp16(-1.0)];\n")
	b.WriteString("        int32 rpax = const()[name=string(\"rpax\"), val=int32(3)];\n")
	b.WriteString("        bool rpil = const()[name=string(\"rpil\"), val=bool(false)];\n")
	// RoPE Q
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,2]> q_pair = reshape(shape=rp_sh,x=q)[name=string(\"q_pair\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> q_e = slice_by_size(x=q_pair,begin=rp_b0,size=rp_s1)[name=string(\"q_e\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> q_o = slice_by_size(x=q_pair,begin=rp_b1,size=rp_s1)[name=string(\"q_o\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> q_no = mul(x=q_o,y=neg1)[name=string(\"q_no\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,2]> q_rotp = concat(axis=rpax,interleave=rpil,values=(q_no,q_e))[name=string(\"q_rotp\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q_rot = reshape(shape=rp_bk,x=q_rotp)[name=string(\"q_rot\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q_c = mul(x=q,y=rope_cos)[name=string(\"q_c\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q_rs = mul(x=q_rot,y=rope_sin)[name=string(\"q_rs\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q_rope = add(x=q_c,y=q_rs)[name=string(\"q_rope\")];\n", heads, seq, headDim)
	// RoPE K
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,2]> k_pair = reshape(shape=rp_sh,x=k)[name=string(\"k_pair\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> k_e = slice_by_size(x=k_pair,begin=rp_b0,size=rp_s1)[name=string(\"k_e\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> k_o = slice_by_size(x=k_pair,begin=rp_b1,size=rp_s1)[name=string(\"k_o\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> k_no = mul(x=k_o,y=neg1)[name=string(\"k_no\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,2]> k_rotp = concat(axis=rpax,interleave=rpil,values=(k_no,k_e))[name=string(\"k_rotp\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_rot = reshape(shape=rp_bk,x=k_rotp)[name=string(\"k_rot\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_c = mul(x=k,y=rope_cos)[name=string(\"k_c\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_rs = mul(x=k_rot,y=rope_sin)[name=string(\"k_rs\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_rope = add(x=k_c,y=k_rs)[name=string(\"k_rope\")];\n", heads, seq, headDim)
	// SDPA
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q_rope,y=k_rope)[name=string(\"sc1\")];\n", heads, seq, seq)
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
	appendDynamicMatmulFP16(&b, "oo", "af", "Wo", dim, dim, seq)
	// Attention residual: x2 = x + oo
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> x2 = add(x=x,y=oo)[name=string(\"x2\")];\n", dim, seq)

	// === FFN BLOCK ===
	appendDynamicRMSNormFP16(&b, "xn2", "x2", "rw2", dim, seq)
	appendDynamicMatmulFP16(&b, "h1", "xn2", "W1", dim, hidden, seq)
	appendDynamicMatmulFP16(&b, "h3", "xn2", "W3", dim, hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> silu_h1 = silu(x=h1)[name=string(\"silu_h1\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu_h1,y=h3)[name=string(\"gate\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> gate_shape = const()[name=string(\"gate_shape\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> gate2 = reshape(shape=gate_shape,x=gate)[name=string(\"gate2\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> gate3 = transpose(perm=pm,x=gate2)[name=string(\"gate3\")];\n", seq, hidden)
	fmt.Fprintf(&b, "        tensor<int32, [4]> w2_shape = const()[name=string(\"w2_shape\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", dim, hidden)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> W22 = reshape(shape=w2_shape,x=W2r)[name=string(\"W22\")];\n", dim, hidden)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> W2 = transpose(perm=pm,x=W22)[name=string(\"W2\")];\n", hidden, dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> ffm = matmul(transpose_x=bF,transpose_y=bF,x=gate3,y=W2)[name=string(\"ffm\")];\n", seq, dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> fft = transpose(perm=pm,x=ffm)[name=string(\"fft\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> ff_shape = const()[name=string(\"ff_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> ff = reshape(shape=ff_shape,x=fft)[name=string(\"ff\")];\n", dim, seq)
	// FFN residual: out = x2 + ff
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> out = add(x=x2,y=ff)[name=string(\"out\")];\n", dim, seq)
	b.WriteString("    } -> (out);\n}\n")
	return b.String()
}

// GenStoriesSDPAForwardInference generates an inference-only attention kernel.
// Same as DynamicTaps but with residual scale baked in and no taps output.
// Output: x + scale * Wo@attn (just dim*seq channels).
func GenStoriesSDPAForwardInference(dim, heads, seq int, residualScale float64) string {
	headDim := dim / heads
	scale := 1.0 / math.Sqrt(float64(headDim))
	spatial := seq + 1 + 4*dim

	var b strings.Builder
	fmt.Fprintf(&b, "%s{\n", buildInfoHeader)
	fmt.Fprintf(&b, "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> inp) {\n", dim, spatial)
	b.WriteString("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n")
	b.WriteString("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n")
	b.WriteString("        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n")
	b.WriteString("        tensor<int32, [4]> x_begin = const()[name=string(\"x_begin\"), val=tensor<int32, [4]>([0,0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [4]> x_size = const()[name=string(\"x_size\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> x = slice_by_size(x=inp,begin=x_begin,size=x_size)[name=string(\"x\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rw_begin = const()[name=string(\"rw_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rw_size = const()[name=string(\"rw_size\"), val=tensor<int32, [4]>([1,%d,1,1])];\n", dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,1]> rw = slice_by_size(x=inp,begin=rw_begin,size=rw_size)[name=string(\"rw\")];\n", dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wq_begin = const()[name=string(\"wq_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq+1)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wt_size = const()[name=string(\"wt_size\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wq = slice_by_size(x=inp,begin=wq_begin,size=wt_size)[name=string(\"Wq\")];\n", dim, dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wk_begin = const()[name=string(\"wk_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq+1+dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wk = slice_by_size(x=inp,begin=wk_begin,size=wt_size)[name=string(\"Wk\")];\n", dim, dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wv_begin = const()[name=string(\"wv_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq+1+2*dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wv = slice_by_size(x=inp,begin=wv_begin,size=wt_size)[name=string(\"Wv\")];\n", dim, dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> wo_begin = const()[name=string(\"wo_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq+1+3*dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> Wo = slice_by_size(x=inp,begin=wo_begin,size=wt_size)[name=string(\"Wo\")];\n", dim, dim)
	appendDynamicRMSNormFP16(&b, "xn", "x", "rw", dim, seq)
	appendDynamicMatmulFP16(&b, "qf", "xn", "Wq", dim, dim, seq)
	appendDynamicMatmulFP16(&b, "kf", "xn", "Wk", dim, dim, seq)
	appendDynamicMatmulFP16(&b, "vf", "xn", "Wv", dim, dim, seq)
	// Reshape for multi-head attention
	fmt.Fprintf(&b, "        tensor<int32, [4]> q_shape = const()[name=string(\"q_shape\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=q_shape,x=qf)[name=string(\"q4\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=string(\"q\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=q_shape,x=kf)[name=string(\"k4\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=k4)[name=string(\"k\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=q_shape,x=vf)[name=string(\"v4\")];\n", heads, headDim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=v4)[name=string(\"v\")];\n", heads, seq, headDim)
	// RoPE
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> rope_cos = const()[name=string(\"rope_cos\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/rope_cos.bin\"), offset=uint64(64)))];\n", seq, headDim, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> rope_sin = const()[name=string(\"rope_sin\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/rope_sin.bin\"), offset=uint64(64)))];\n", seq, headDim, seq, headDim)
	pairs := seq * headDim / 2
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_sh = const()[name=string(\"rp_sh\"), val=tensor<int32, [4]>([1,%d,%d,2])];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_s1 = const()[name=string(\"rp_s1\"), val=tensor<int32, [4]>([1,%d,%d,1])];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rp_bk = const()[name=string(\"rp_bk\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", heads, seq, headDim)
	b.WriteString("        tensor<int32, [4]> rp_b0 = const()[name=string(\"rp_b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n")
	b.WriteString("        tensor<int32, [4]> rp_b1 = const()[name=string(\"rp_b1\"), val=tensor<int32, [4]>([0,0,0,1])];\n")
	b.WriteString("        fp16 neg1 = const()[name=string(\"neg1\"), val=fp16(-1.0)];\n")
	b.WriteString("        int32 rpax = const()[name=string(\"rpax\"), val=int32(3)];\n")
	b.WriteString("        bool rpil = const()[name=string(\"rpil\"), val=bool(false)];\n")
	// Apply RoPE to Q
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,2]> q_pair = reshape(shape=rp_sh,x=q)[name=string(\"q_pair\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> q_e = slice_by_size(x=q_pair,begin=rp_b0,size=rp_s1)[name=string(\"q_e\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> q_o = slice_by_size(x=q_pair,begin=rp_b1,size=rp_s1)[name=string(\"q_o\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> q_no = mul(x=q_o,y=neg1)[name=string(\"q_no\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,2]> q_rotp = concat(axis=rpax,interleave=rpil,values=(q_no,q_e))[name=string(\"q_rotp\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q_rot = reshape(shape=rp_bk,x=q_rotp)[name=string(\"q_rot\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q_c = mul(x=q,y=rope_cos)[name=string(\"q_c\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q_rs = mul(x=q_rot,y=rope_sin)[name=string(\"q_rs\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> q_rope = add(x=q_c,y=q_rs)[name=string(\"q_rope\")];\n", heads, seq, headDim)
	// Apply RoPE to K
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,2]> k_pair = reshape(shape=rp_sh,x=k)[name=string(\"k_pair\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> k_e = slice_by_size(x=k_pair,begin=rp_b0,size=rp_s1)[name=string(\"k_e\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> k_o = slice_by_size(x=k_pair,begin=rp_b1,size=rp_s1)[name=string(\"k_o\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,1]> k_no = mul(x=k_o,y=neg1)[name=string(\"k_no\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,2]> k_rotp = concat(axis=rpax,interleave=rpil,values=(k_no,k_e))[name=string(\"k_rotp\")];\n", heads, pairs)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_rot = reshape(shape=rp_bk,x=k_rotp)[name=string(\"k_rot\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_c = mul(x=k,y=rope_cos)[name=string(\"k_c\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_rs = mul(x=k_rot,y=rope_sin)[name=string(\"k_rs\")];\n", heads, seq, headDim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> k_rope = add(x=k_c,y=k_rs)[name=string(\"k_rope\")];\n", heads, seq, headDim)
	// Scaled dot-product attention
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q_rope,y=k_rope)[name=string(\"sc1\")];\n", heads, seq, seq)
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
	// Residual with scale baked in: out = (1-scale)*x + scale*oo
	fmt.Fprintf(&b, "        fp16 rs = const()[name=string(\"rs\"), val=fp16(%f)];\n", residualScale)
	fmt.Fprintf(&b, "        fp16 rs1 = const()[name=string(\"rs1\"), val=fp16(%f)];\n", 1.0-residualScale)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> oo_s = mul(x=oo,y=rs)[name=string(\"oo_s\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> x_s = mul(x=x,y=rs1)[name=string(\"x_s\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> out = add(x=x_s,y=oo_s)[name=string(\"out\")];\n", dim, seq)
	b.WriteString("    } -> (out);\n}\n")
	return b.String()
}

// GenStoriesFFNForwardInference generates an inference-only FFN kernel.
// Same as DynamicTaps but with residual scale baked in and no taps output.
// Output: x2 + scale * W2@gate (just dim*seq channels).
func GenStoriesFFNForwardInference(dim, hidden, seq int, residualScale float64) string {
	spatial := seq + 1 + 3*hidden

	var b strings.Builder
	fmt.Fprintf(&b, "%s{\n", buildInfoHeader)
	fmt.Fprintf(&b, "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> inp) {\n", dim, spatial)
	b.WriteString("        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n")
	b.WriteString("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n")
	b.WriteString("        tensor<int32, [4]> x_begin = const()[name=string(\"x_begin\"), val=tensor<int32, [4]>([0,0,0,0])];\n")
	fmt.Fprintf(&b, "        tensor<int32, [4]> x_size = const()[name=string(\"x_size\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> x2 = slice_by_size(x=inp,begin=x_begin,size=x_size)[name=string(\"x2\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rw_begin = const()[name=string(\"rw_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> rw_size = const()[name=string(\"rw_size\"), val=tensor<int32, [4]>([1,%d,1,1])];\n", dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,1]> rw = slice_by_size(x=inp,begin=rw_begin,size=rw_size)[name=string(\"rw\")];\n", dim)
	fmt.Fprintf(&b, "        tensor<int32, [4]> w_shape = const()[name=string(\"w_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, hidden)
	fmt.Fprintf(&b, "        tensor<int32, [4]> w1_begin = const()[name=string(\"w1_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq+1)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> W1 = slice_by_size(x=inp,begin=w1_begin,size=w_shape)[name=string(\"W1\")];\n", dim, hidden)
	fmt.Fprintf(&b, "        tensor<int32, [4]> w3_begin = const()[name=string(\"w3_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq+1+hidden)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> W3 = slice_by_size(x=inp,begin=w3_begin,size=w_shape)[name=string(\"W3\")];\n", dim, hidden)
	fmt.Fprintf(&b, "        tensor<int32, [4]> w2_begin = const()[name=string(\"w2_begin\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq+1+2*hidden)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> W2r = slice_by_size(x=inp,begin=w2_begin,size=w_shape)[name=string(\"W2r\")];\n", dim, hidden)
	appendDynamicRMSNormFP16(&b, "xn", "x2", "rw", dim, seq)
	appendDynamicMatmulFP16(&b, "h1", "xn", "W1", dim, hidden, seq)
	appendDynamicMatmulFP16(&b, "h3", "xn", "W3", dim, hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> silu_h1 = silu(x=h1)[name=string(\"silu_h1\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu_h1,y=h3)[name=string(\"gate\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> gate_shape = const()[name=string(\"gate_shape\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> gate2 = reshape(shape=gate_shape,x=gate)[name=string(\"gate2\")];\n", hidden, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> gate3 = transpose(perm=pm,x=gate2)[name=string(\"gate3\")];\n", seq, hidden)
	fmt.Fprintf(&b, "        tensor<int32, [4]> w2_shape = const()[name=string(\"w2_shape\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", dim, hidden)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> W22 = reshape(shape=w2_shape,x=W2r)[name=string(\"W22\")];\n", dim, hidden)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> W2 = transpose(perm=pm,x=W22)[name=string(\"W2\")];\n", hidden, dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> ffm = matmul(transpose_x=bF,transpose_y=bF,x=gate3,y=W2)[name=string(\"ffm\")];\n", seq, dim)
	fmt.Fprintf(&b, "        tensor<fp16, [1,1,%d,%d]> fft = transpose(perm=pm,x=ffm)[name=string(\"fft\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<int32, [4]> ff_shape = const()[name=string(\"ff_shape\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> ff = reshape(shape=ff_shape,x=fft)[name=string(\"ff\")];\n", dim, seq)
	// Residual with scale baked in: out = x2 + scale * ff
	fmt.Fprintf(&b, "        fp16 rs = const()[name=string(\"rs\"), val=fp16(%f)];\n", residualScale)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> ff_s = mul(x=ff,y=rs)[name=string(\"ff_s\")];\n", dim, seq)
	fmt.Fprintf(&b, "        tensor<fp16, [1,%d,1,%d]> out = add(x=x2,y=ff_s)[name=string(\"out\")];\n", dim, seq)
	b.WriteString("    } -> (out);\n}\n")
	return b.String()
}
