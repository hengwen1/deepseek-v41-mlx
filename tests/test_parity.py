"""Tiny-config numeric parity: deepseek_v41_mlx vs DeepSeek's reference model.py.

The reference (docs/reference/model.py) runs as shipped on CPU in fp32; only its
TileLang kernels are substituted by transcriptions written from their published
semantics (tests/kernel_stub.py), plus three declared adaptations:

* ``ParallelEngramEmbedding.forward`` keeps fp32 instead of casting to bf16
  (the cast is lossless there — fp8 values times power-of-two scales — but the
  bf16 tensor would crash torch's fp32 matmul);
* ``ParallelHead.forward`` returns all positions, not just the last;
* ``Indexer.forward`` reads the *owner's own* key cache (``self.k_cache``)
  instead of the process-global pointer. Without this the reference's decode
  reads the LAST owner's cache whenever a ratio-2 owner's group is incomplete —
  a reference artifact this test also measures (see docs/upstream-notes.md).
  Prefill from 0 is unaffected.

Tie-breaking note: indexer scores are ReLU'd, so exact zero ties are possible;
when more visible columns tie at the top-k boundary than fit, torch and MLX may
legitimately pick different sets. The tiny config uses 8 index heads (zero
probability ~2^-8 per entry) and a fixed seed for which no tie lands on the
boundary; a genuine regression still fails loudly.

The config exercises every fragile path: hc mixing + staggered pre-chaining,
engram at layers 1 and 2 (with pad-token and sequence-start boundaries),
ratio-2 and ratio-1 compression, cross-layer KV/index sharing (consumers after
their producers), an index source that owns no keys (layer 7), two-level
candidate selection (source 5, consumer 7), attention sinks, rope-inverse,
clamped SwiGLU, 8 routed experts top-3, and sequences long enough that the
indexer genuinely selects (index_topk 4 < nb up to 28).

    .venv/bin/python tests/test_parity.py
"""

import json
import os
import re
import sys
import types
import importlib.util

import numpy as np
import torch

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "reference"))

# ---- stub the modules the reference imports but this test never exercises ----
_pil = types.ModuleType("PIL")
_pil.Image = types.SimpleNamespace()
_pil.ImageOps = types.SimpleNamespace()
sys.modules.setdefault("PIL", _pil)

import kernel_stub
sys.modules["kernel"] = kernel_stub

import engram as REF_engram  # noqa: E402  (docs/reference/engram.py)
import model as REF          # noqa: E402  (docs/reference/model.py)

import mlx.core as mx        # noqa: E402

from deepseek_v41_mlx.config import ModelArgs   # noqa: E402
from deepseek_v41_mlx.model import Model        # noqa: E402


# --------------------------------------------------------------------------
# tiny config
# --------------------------------------------------------------------------

B, S, DECODE = 2, 48, 8
MAXSEQ = 64
VOCAB = 96

CFG = dict(
    vocab_size=VOCAB, dim=64, n_layers=8, moe_inter_dim=32,
    n_heads=4, n_routed_experts=8, n_shared_experts=1, n_activated_experts=3,
    score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=1.0,
    q_lora_rank=32, head_dim=64, rope_head_dim=16, norm_eps=1e-20,
    o_groups=4, o_lora_rank=16, window_size=8,
    compress_ratios=(0, 0, 2, 2, 2, 1, 1, 1),
    kv_source_layers=(2, 5), index_source_layers=(2, 5, 7),
    candidate_source_layer=5, candidate_topk_blocks=3, candidate_block_size=2,
    original_seq_len=64, rope_theta=10000.0, rope_factor=4.0, beta_fast=32, beta_slow=1,
    compress_rope_theta=40000.0,
    index_n_heads=8, index_head_dim=32, index_topk=4,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    engram_layer_ids=(1, 2), engram_num_embeddings=(384, 552),
    engram_max_ngram_size=4, engram_vocab_size=50,
    engram_n_heads=2, engram_head_dim=32,
    engram_compressed_vocab_size=VOCAB,
)


# --------------------------------------------------------------------------
# reference adaptations (declared in the module docstring)
# --------------------------------------------------------------------------

REF_engram.build_compressed_token_map = lambda tok: (list(range(VOCAB)), VOCAB)


def _pee_forward_fp32(self, indices):
    values = torch.nn.functional.embedding(indices, self.weight)
    scales = torch.nn.functional.embedding(indices, self.scale)
    values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
    return values.flatten(-2)


REF.ParallelEngramEmbedding.forward = _pee_forward_fp32

_orig_head_forward = REF.ParallelHead.forward
REF.ParallelHead.forward = lambda self, x, full_logits=False: _orig_head_forward(self, x, True)

_ORIG_INDEXER_FORWARD = REF.Indexer.forward


def _own_key_indexer_forward(self, x, qr, latent, start_pos, offset):
    if self.owns_k:
        REF.shared_attn.index_k = self.k_cache
    return _ORIG_INDEXER_FORWARD(self, x, qr, latent, start_pos, offset)


def patch_indexer(on: bool):
    REF.Indexer.forward = _own_key_indexer_forward if on else _ORIG_INDEXER_FORWARD


patch_indexer(True)


# --------------------------------------------------------------------------
# weight transplant
# --------------------------------------------------------------------------

def transplant(rm, mm, rng):
    D = CFG["dim"]

    def w(*shape, s=None):
        fan = shape[-1] if len(shape) > 1 else shape[0]
        s = s if s is not None else fan ** -0.5
        return (rng.standard_normal(shape) * s).astype(np.float32)

    def setp(param, arr):
        with torch.no_grad():
            param.copy_(torch.from_numpy(np.asarray(arr)))

    def setl(rl, ml, W):
        setp(rl.weight, W)
        ml.weight = mx.array(W)

    def setnorm(rn, mn, dim):
        g = (1.0 + 0.1 * rng.standard_normal(dim)).astype(np.float32)
        setp(rn.weight, g)
        mn.weight = mx.array(g)

    setl(rm.embed, mm.embed, w(VOCAB, D, s=0.5))
    setl(rm.head, mm.head, w(VOCAB, D, s=0.3))
    setnorm(rm.norm, mm.norm, D)

    for li in range(CFG["n_layers"]):
        rb, mb = rm.layers[li], mm.layers[li]
        for nm in ("attn", "ffn"):
            fn = w((2 + 4) * 4, 4 * D, s=0.05)
            ba = (0.2 * rng.standard_normal((2 + 4) * 4)).astype(np.float32)
            sc = (0.5 + 0.2 * rng.standard_normal(3)).astype(np.float32)
            setp(getattr(rb, f"hc_{nm}_fn"), fn); setattr(mb, f"hc_{nm}_fn", mx.array(fn))
            setp(getattr(rb, f"hc_{nm}_base"), ba); setattr(mb, f"hc_{nm}_base", mx.array(ba))
            setp(getattr(rb, f"hc_{nm}_scale"), sc); setattr(mb, f"hc_{nm}_scale", mx.array(sc))
        setnorm(rb.attn_norm, mb.attn_norm, D)
        setnorm(rb.ffn_norm, mb.ffn_norm, D)

        ra, ma = rb.attn, mb.attn
        NH, HD = CFG["n_heads"], CFG["head_dim"]
        QLR, OLR, G = CFG["q_lora_rank"], CFG["o_lora_rank"], CFG["o_groups"]
        setl(ra.wq_a, ma.wq_a, w(QLR, D))
        setl(ra.wq_b, ma.wq_b, w(NH * HD, QLR))
        setl(ra.wkv, ma.wkv, w(HD, D))
        setl(ra.wo_a, ma.wo_a, w(G * OLR, NH * HD // G))
        setl(ra.wo_b, ma.wo_b, w(D, G * OLR))
        setnorm(ra.q_norm, ma.q_norm, QLR)
        setnorm(ra.kv_norm, ma.kv_norm, HD)
        sk = w(NH, s=0.5)
        setp(ra.attn_sink, sk); ma.attn_sink = mx.array(sk)

        if ra.compressor is not None:
            setl(ra.compressor.wkv, ma.compressor.wkv, w(HD, D))
            if CFG["compress_ratios"][li] > 1:
                setl(ra.compressor.wgate, ma.compressor.wgate, w(HD, D))
            setnorm(ra.compressor.norm, ma.compressor.norm, HD)
        if ra.indexer is not None:
            INH, IHD = CFG["index_n_heads"], CFG["index_head_dim"]
            setl(ra.indexer.wq_b, ma.indexer.wq_b, w(INH * IHD, QLR))
            setl(ra.indexer.weights_proj, ma.indexer.weights_proj, w(INH, D))
            if ra.indexer.owns_k:
                setl(ra.indexer.wk, ma.indexer.wk, w(IHD, HD))
                setnorm(ra.indexer.k_norm, ma.indexer.k_norm, IHD)

        NE, INTER = CFG["n_routed_experts"], CFG["moe_inter_dim"]
        gw = w(NE, D)
        setp(rb.ffn.gate.weight, gw); mb.ffn.gate.weight = mx.array(gw)
        gb = (0.3 * rng.standard_normal(NE)).astype(np.float32)
        setp(rb.ffn.gate.bias, gb); mb.ffn.gate.bias = mx.array(gb)
        mb.ffn.gate.bias_vl = mx.zeros((NE,), dtype=mx.float32)  # unused, text-only

        e1, e2, e3 = w(NE, INTER, D), w(NE, D, INTER), w(NE, INTER, D)
        for i in range(NE):
            setp(rb.ffn.experts[i].w1.weight, e1[i])
            setp(rb.ffn.experts[i].w2.weight, e2[i])
            setp(rb.ffn.experts[i].w3.weight, e3[i])
        mb.ffn.experts.gate_proj.weight = mx.array(e1)
        mb.ffn.experts.down_proj.weight = mx.array(e2)
        mb.ffn.experts.up_proj.weight = mx.array(e3)
        setl(rb.ffn.shared_experts.w1, mb.ffn.shared_experts.w1, w(INTER, D))
        setl(rb.ffn.shared_experts.w2, mb.ffn.shared_experts.w2, w(D, INTER))
        setl(rb.ffn.shared_experts.w3, mb.ffn.shared_experts.w3, w(INTER, D))

        if rb.engram is not None:
            re, me = rb.engram, mb.engram
            rows = CFG["engram_num_embeddings"][re.layer_hash_index]
            EHD = CFG["engram_head_dim"]
            vals = torch.from_numpy(w(rows, EHD, s=0.5)).to(torch.float8_e4m3fn).float().numpy()
            scales = (2.0 ** rng.integers(-2, 3, size=(rows, EHD // 32))).astype(np.float32)
            setp(re.embed.weight, vals); me.embed.weight = mx.array(vals)
            setp(re.embed.scale, scales); me.embed.scale = mx.array(scales)
            n_cols = (CFG["engram_max_ngram_size"] - 1) * CFG["engram_n_heads"]
            setl(re.wkv, me.wkv, w(D * 5, n_cols * EHD))
            qw = (1.0 + 0.3 * rng.standard_normal((4, D))).astype(np.float32)
            kw = (1.0 + 0.3 * rng.standard_normal((4, D))).astype(np.float32)
            setp(re.q_weight, qw); me.q_weight = mx.array(qw)
            setp(re.k_weight, kw); me.k_weight = mx.array(kw)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def rel_max_diff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-9)


def ref_prefill(rm, ids_np, n):
    with torch.no_grad():
        _, logits, _ = rm(torch.from_numpy(ids_np[:, :n]), 0)
    return logits.float().numpy()          # [B, n, V] (head patched to full logits)


def ours_prefill(mm, ids_np, n, chunks=None):
    cache = mm.make_cache(bsz=B, max_seq_len=MAXSEQ)
    if chunks is None:
        return np.array(mm(mx.array(ids_np[:, :n]), cache)), cache
    outs = []
    for a, b in chunks:
        outs.append(np.array(mm(mx.array(ids_np[:, a:b]), cache)))
    return np.concatenate(outs, axis=1), cache


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def set_fake_quant(enabled: bool):
    import deepseek_v41_mlx.fakequant as FQ
    FQ.DISABLE = not enabled
    kernel_stub.DISABLE_FAKE_QUANT = not enabled


def battery(rm, mm, ids, results, tag, thr):
    """The full comparison battery at the current fake-quant setting."""
    # ---- full prefill parity at several lengths (odd lengths exercise the
    # ratio-2 remainder path; 7 < window exercises the short-sequence path) ----
    for n in (7, 33, 48):
        r = ref_prefill(rm, ids, n)
        m, _ = ours_prefill(mm, ids, n)
        d = rel_max_diff(m, r)
        results[f"[{tag}] prefill[{n}] all-position"] = d
        assert d < thr, f"[{tag}] prefill length {n}: rel diff {d}"

    r48 = ref_prefill(rm, ids, S)   # leaves the reference cache at offset 48

    # ---- reference decode (patched to sane index keys) vs our decode,
    # and both vs the reference's own full prefill of S+DECODE ----
    ref_dec = []
    with torch.no_grad():
        for i in range(S, S + DECODE):
            _, lg, _ = rm(torch.from_numpy(ids[:, i:i + 1]), i)
            ref_dec.append(lg.float().numpy())
    ref_dec = np.concatenate(ref_dec, axis=1)                     # [B, DECODE, V]

    m48, cache = ours_prefill(mm, ids, S)
    our_dec = []
    for i in range(S, S + DECODE):
        our_dec.append(np.array(mm(mx.array(ids[:, i:i + 1]), cache)))
    our_dec = np.concatenate(our_dec, axis=1)

    ref_full = ref_prefill(rm, ids, S + DECODE)
    d = rel_max_diff(our_dec, ref_dec)
    results[f"[{tag}] decode vs reference decode (index-k patched)"] = d
    assert d < thr, f"[{tag}] decode parity {d}"
    d = rel_max_diff(our_dec, ref_full[:, S:])
    results[f"[{tag}] our decode vs reference FULL prefill"] = d
    assert d < thr, f"[{tag}] decode vs full prefill {d}"

    # ---- chunked prefill == single forward (ours; reference has no chunked path) ----
    mfull, _ = ours_prefill(mm, ids, S + DECODE)
    mchunk, _ = ours_prefill(mm, ids, S + DECODE,
                             chunks=[(0, 13), (13, 14), (14, 31), (31, 48), (48, 56)])
    d = rel_max_diff(mchunk, mfull)
    results[f"[{tag}] chunked prefill vs single forward (ours)"] = d
    assert d < thr, f"[{tag}] chunked prefill {d}"
    d = rel_max_diff(mfull, ref_full)
    results[f"[{tag}] our 56-token prefill vs reference"] = d
    assert d < thr
    return r48, ref_dec, mfull, ref_full


def fake_quant_unit_parity():
    """The three QAT-sim paths must agree BIT-EXACTLY between the torch stub and
    the MLX port — including exact rounding midpoints and power-of-two amax
    boundaries. This is what makes the looser end-to-end threshold of the
    fake-quant-on pass principled: any e2e gap comes from ~1e-7 *upstream*
    noise landing on a shared, correctly-implemented rounding boundary."""
    rng = np.random.default_rng(3)
    xs = [
        (rng.standard_normal((256, 64)) * np.exp(rng.standard_normal((256, 1)))).astype(np.float32),
        # exact e4m3/e2m1 midpoints at power-of-two scales
        (np.array([0.033203125, -0.033203125, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5,
                   5.0, -5.0, 448.0, 6.0, 1e-9, 0.0, -0.0, 2.0] * 8, dtype=np.float32)
         .reshape(2, 64) * 2.0 ** rng.integers(-8, 9, size=(2, 1)).astype(np.float32)),
    ]
    import deepseek_v41_mlx.fakequant as FQ
    for x in xs:
        t = kernel_stub.act_quant(torch.from_numpy(x.copy()), 32, "ue8m0",
                                  torch.float8_e8m0fnu, True).numpy()
        m = np.array(FQ.fake_quant_fp8_ue8m0(mx.array(x), 32))
        assert np.array_equal(t, m), "fp8/ue8m0 fake-quant not bit-exact"
        t = kernel_stub.fp4_act_quant(torch.from_numpy(x.copy()), 32, True).numpy()
        m = np.array(FQ.fake_quant_fp4_ue8m0(mx.array(x), 32))
        assert np.array_equal(t, m), "fp4/ue8m0 fake-quant not bit-exact"
        t = kernel_stub.fp4_act_quant(torch.from_numpy(x.copy()), 16, True,
                                      torch.float8_e4m3fn).numpy()
        m = np.array(FQ.fake_quant_fp4_e4m3(mx.array(x), 16))
        assert np.array_equal(t, m), "fp4/e4m3 fake-quant not bit-exact"
    print("  fake-quant unit parity: bit-exact (fp8/ue8m0, fp4/ue8m0, fp4/e4m3)")


def dequant_unit_parity():
    """Checkpoint decode: synthetic packed fp8/fp4/engram tensors, torch-side
    reference dequant (transcribed from convert.py / fp8_gemm / fp4_gemm scale
    semantics, using the reference FP4_TABLE nibble order) vs ours — bit-exact."""
    sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda x, **k: x,
                                                         trange=lambda *a, **k: range(*a)))
    import convert as REF_CONVERT  # docs/reference/convert.py — FP4_TABLE truth
    from deepseek_v41_mlx.dequant import dequant_fp4, dequant_fp8, dequant_fp8_rows

    rng = np.random.default_rng(11)
    OUT, IN = 96, 64                                  # non-multiple of 32 on out

    # fp8 e4m3 [out, in] + ue8m0 [ceil(out/32), ceil(in/32)]
    wb = rng.integers(0, 256, size=(OUT, IN), dtype=np.uint8)
    wb[wb % 128 == 127] = 0                           # avoid e4m3 NaN codes
    sb = rng.integers(118, 137, size=(3, 2), dtype=np.uint8)
    tw = torch.from_numpy(wb).view(torch.float8_e4m3fn).float()
    ts = torch.from_numpy(2.0 ** (sb.astype(np.float64) - 127)).float()
    ref = (tw * ts.repeat_interleave(32, 0)[:OUT].repeat_interleave(32, 1)[:, :IN])
    ours = np.array(dequant_fp8(mx.array(wb), mx.array(sb), mx.float32))
    assert np.array_equal(ref.numpy(), ours), "fp8 dequant not bit-exact"

    # fp4 e2m1 packed [out, in//2] + ue8m0 [out, in//32]
    pb = rng.integers(0, 256, size=(OUT, IN // 2), dtype=np.uint8)
    s4 = rng.integers(118, 137, size=(OUT, IN // 32), dtype=np.uint8)
    lo, hi = pb & 0x0F, (pb >> 4) & 0x0F
    vals = REF_CONVERT.FP4_TABLE[torch.from_numpy(np.stack([lo, hi], -1)).long()].flatten(1)
    ts4 = torch.from_numpy(2.0 ** (s4.astype(np.float64) - 127)).float()
    ref4 = vals * ts4.repeat_interleave(32, 1)[:, :IN]
    ours4 = np.array(dequant_fp4(mx.array(pb.view(np.int8)), mx.array(s4), mx.float32))
    assert np.array_equal(ref4.numpy(), ours4), "fp4 dequant not bit-exact"

    # engram-row layout: e4m3 [rows, d] + ue8m0 [rows, d//32]
    er = torch.from_numpy(wb).view(torch.float8_e4m3fn).float()
    es = rng.integers(118, 137, size=(OUT, IN // 32), dtype=np.uint8)
    tse = torch.from_numpy(2.0 ** (es.astype(np.float64) - 127)).float()
    refr = er * tse.repeat_interleave(32, 1)
    oursr = np.array(dequant_fp8_rows(mx.array(wb), mx.array(es)))
    assert np.array_equal(refr.numpy(), oursr), "engram-row dequant not bit-exact"
    print("  dequant unit parity: bit-exact (fp8 32x32, fp4 packed, engram rows)")


def _write_tiny_release(src: str, rng):
    """A synthetic checkpoint in the raw release layout: inference-style names,
    fp8 weights + ue8m0 scales, fp4-packed experts, fp8 engram tables, bf16
    everything else, plus mtp.* and the vision tower, sharded with an index."""
    from safetensors.torch import save_file

    D, HD, QLR = CFG["dim"], CFG["head_dim"], CFG["q_lora_rank"]
    NH, G, OLR = CFG["n_heads"], CFG["o_groups"], CFG["o_lora_rank"]
    INH, IHD = CFG["index_n_heads"], CFG["index_head_dim"]
    NE, INTER = CFG["n_routed_experts"], CFG["moe_inter_dim"]
    t = {}

    def fp8(name, o, i):
        t[name + ".weight"] = torch.from_numpy(
            (rng.standard_normal((o, i)) * 0.2).astype(np.float32)).to(torch.float8_e4m3fn)
        t[name + ".scale"] = torch.from_numpy(rng.integers(
            122, 130, size=((o + 31) // 32, (i + 31) // 32), dtype=np.uint8)
        ).view(torch.float8_e8m0fnu)

    def fp4(name, o, i):
        t[name + ".weight"] = torch.from_numpy(
            rng.integers(-128, 128, size=(o, i // 2), dtype=np.int8))
        t[name + ".scale"] = torch.from_numpy(rng.integers(
            122, 130, size=(o, i // 32), dtype=np.uint8)).view(torch.float8_e8m0fnu)

    def bf16(name, *shape, scale=0.2):
        t[name] = torch.from_numpy(
            (rng.standard_normal(shape) * scale).astype(np.float32)).to(torch.bfloat16)

    def f32(name, *shape, scale=0.2):
        t[name] = torch.from_numpy((rng.standard_normal(shape) * scale).astype(np.float32))

    bf16("embed.weight", VOCAB, D)
    bf16("head.weight", VOCAB, D)
    bf16("norm.weight", D)

    for li in range(CFG["n_layers"]):
        p = f"layers.{li}."
        fp8(p + "attn.wq_a", QLR, D)
        fp8(p + "attn.wq_b", NH * HD, QLR)
        fp8(p + "attn.wkv", HD, D)
        fp8(p + "attn.wo_a", G * OLR, NH * HD // G)
        fp8(p + "attn.wo_b", D, G * OLR)
        bf16(p + "attn.q_norm.weight", QLR, scale=1.0)
        bf16(p + "attn.kv_norm.weight", HD, scale=1.0)
        f32(p + "attn.attn_sink", NH, scale=0.5)
        bf16(p + "attn_norm.weight", D, scale=1.0)
        bf16(p + "ffn_norm.weight", D, scale=1.0)
        for nm in ("attn", "ffn"):
            f32(p + f"hc_{nm}_fn", 24, 4 * D, scale=0.05)
            f32(p + f"hc_{nm}_base", 24)
            f32(p + f"hc_{nm}_scale", 3, scale=0.5)
        bf16(p + "ffn.gate.weight", NE, D)
        f32(p + "ffn.gate.bias", NE, scale=0.3)
        f32(p + "ffn.gate.bias_vl", NE, scale=0.3)
        for w, o, i in (("w1", INTER, D), ("w2", D, INTER), ("w3", INTER, D)):
            fp8(p + f"ffn.shared_experts.{w}", o, i)
            for e in range(NE):
                fp4(p + f"ffn.experts.{e}.{w}", o, i)
        if li in CFG["kv_source_layers"]:
            bf16(p + "attn.compressor.wkv.weight", HD, D)
            if CFG["compress_ratios"][li] > 1:
                bf16(p + "attn.compressor.wgate.weight", HD, D)
            bf16(p + "attn.compressor.norm.weight", HD, scale=1.0)
        if li in CFG["index_source_layers"]:
            fp8(p + "attn.indexer.wq_b", INH * IHD, QLR)
            bf16(p + "attn.indexer.weights_proj.weight", INH, D)
            if li in CFG["kv_source_layers"]:
                bf16(p + "attn.indexer.wk.weight", IHD, HD)
                bf16(p + "attn.indexer.k_norm.weight", IHD, scale=1.0)
        if li in CFG["engram_layer_ids"]:
            hi = CFG["engram_layer_ids"].index(li)
            rows, EHD = CFG["engram_num_embeddings"][hi], CFG["engram_head_dim"]
            t[p + "engram.embed.weight"] = torch.from_numpy(
                (rng.standard_normal((rows, EHD)) * 0.3).astype(np.float32)).to(torch.float8_e4m3fn)
            t[p + "engram.embed.scale"] = torch.from_numpy(rng.integers(
                122, 130, size=(rows, EHD // 32), dtype=np.uint8)).view(torch.float8_e8m0fnu)
            cols = (CFG["engram_max_ngram_size"] - 1) * CFG["engram_n_heads"]
            fp8(p + "engram.wkv", D * 5, cols * EHD)
            f32(p + "engram.q_weight", 4, D, scale=1.0)
            f32(p + "engram.k_weight", 4, D, scale=1.0)

    # mtp draft stack (kept by default; keep_mtp=False drops it) and vision
    # tower (always passes through)
    fp8("mtp.0.attn.wq_a", QLR, D)
    fp4("mtp.0.ffn.experts.0.w1", INTER, D)
    f32("mtp.0.hc_attn_base", 24)
    bf16("mtp.2.norm.weight", D, scale=1.0)
    bf16("vision.patch_embed.proj.weight", 32, 3, 14, 14)
    bf16("vision.blocks.0.attn.wqkv.weight", 96, 32)
    bf16("aligner.w1.weight", D, 32)
    bf16("aligner.w1.bias", D)
    bf16("image_start", D)
    bf16("image_end", D)
    bf16("image_newline", D)

    os.makedirs(src, exist_ok=True)
    names = sorted(t)
    half = len(names) // 2
    index = {}
    for i, chunk in enumerate((names[:half], names[half:])):
        fn = f"model-{i:05d}-of-00002.safetensors"
        save_file({k: t[k] for k in chunk}, os.path.join(src, fn))
        for k in chunk:
            index[k] = fn
    json.dump({"metadata": {}, "weight_map": index},
              open(os.path.join(src, "model.safetensors.index.json"), "w"))

    hub_cfg = {
        "model_type": "deepseek_v41", "bos_token_id": 0, "eos_token_id": 1,
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32],
                                "scale_fmt": "ue8m0", "expert_dtype": "fp4"},
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": VOCAB, "hidden_size": CFG["dim"],
            "num_hidden_layers": CFG["n_layers"],
            "moe_intermediate_size": CFG["moe_inter_dim"],
            "num_attention_heads": CFG["n_heads"], "head_dim": CFG["head_dim"],
            "qk_rope_head_dim": CFG["rope_head_dim"],
            "q_lora_rank": CFG["q_lora_rank"], "o_lora_rank": CFG["o_lora_rank"],
            "o_groups": CFG["o_groups"], "sliding_window": CFG["window_size"],
            "rms_norm_eps": CFG["norm_eps"],
            "n_routed_experts": CFG["n_routed_experts"], "n_shared_experts": 1,
            "num_experts_per_tok": CFG["n_activated_experts"],
            "scoring_func": "sqrtsoftplus", "norm_topk_prob": True,
            "routed_scaling_factor": CFG["route_scale"],
            "swiglu_limit": CFG["swiglu_limit"],
            "compress_ratios": list(CFG["compress_ratios"]),
            "compress_rope_theta": CFG["compress_rope_theta"],
            "kv_source_layer_ids": list(CFG["kv_source_layers"]),
            "index_source_layer_ids": list(CFG["index_source_layers"]),
            "index_n_heads": CFG["index_n_heads"],
            "index_head_dim": CFG["index_head_dim"], "index_topk": CFG["index_topk"],
            "candidate_source_layer_id": CFG["candidate_source_layer"],
            "candidate_topk_blocks": CFG["candidate_topk_blocks"],
            "candidate_block_size": CFG["candidate_block_size"],
            "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
            "engram_layer_ids": list(CFG["engram_layer_ids"]),
            "engram_num_embeddings": list(CFG["engram_num_embeddings"]),
            "engram_max_ngram_size": 4, "engram_vocab_size": CFG["engram_vocab_size"],
            "engram_n_heads": CFG["engram_n_heads"],
            "engram_head_dim": CFG["engram_head_dim"],
            "engram_pad_token_id": 2,
            "engram_compressed_vocab_size": VOCAB,
            "rope_theta": CFG["rope_theta"],
            "rope_scaling": {"rope_type": "yarn", "factor": CFG["rope_factor"],
                             "beta_fast": 32, "beta_slow": 1,
                             "original_max_position_embeddings": CFG["original_seq_len"]},
            "max_position_embeddings": 1024,
            "num_nextn_predict_layers": 3,
        },
        "vision_config": {"model_type": "deepseek_v41_vision"},
    }
    json.dump(hub_cfg, open(os.path.join(src, "config.json"), "w"))
    return index


def release_roundtrip():
    """Raw-release-layout checkpoint -> convert -> STRICT load -> forward, at
    bf16 and quantized; full tensor accounting; and the strictness must
    actually trip when a tensor goes missing."""
    import json as _json
    import shutil
    import tempfile

    from deepseek_v41_mlx.convert import convert, parse_bits
    from deepseek_v41_mlx.load import load

    rng = np.random.default_rng(19)
    root = tempfile.mkdtemp(prefix="dsv41_tiny_")
    src = os.path.join(root, "src")
    index = _write_tiny_release(src, rng)

    n_mtp = sum(1 for k in index if k.startswith("mtp.") and not k.endswith(".scale"))
    n_mtp_raw = sum(1 for k in index if k.startswith("mtp."))
    n_vis = sum(1 for k in index if k.startswith(("vision.", "aligner.", "image_")))
    assert n_mtp and n_vis

    # microscaling specs parse (mxf4/mxf8 aliases accepted)
    assert parse_bits("mxfp4") == {"bits": 4, "mode": "mxfp4"}
    assert parse_bits("MXF8") == {"bits": 8, "mode": "mxfp8"}
    assert parse_bits("4") == {"bits": 4, "mode": "affine"}

    accts = {}
    for tag, kw in (("bf16", {}),
                    ("q8_4", dict(bits=8, expert_bits=4, group_size=32)),
                    ("q8_4_e4", dict(bits=8, expert_bits=4, engram_bits=4,
                                     group_size=32)),
                    ("mxfp", dict(bits=parse_bits("mxfp8"),
                                  expert_bits=parse_bits("mxfp4"))),
                    ("omlx4", dict(bits=parse_bits("mxfp4"), omlx_compat=True)),
                    ("nomtp", dict(keep_mtp=False))):
        dst = os.path.join(root, f"dst_{tag}")
        acct = accts[tag] = convert(src, dst, **kw)
        keep = tag != "nomtp"
        assert acct["dropped_mtp"] == (0 if keep else n_mtp_raw), acct
        man_path = os.path.join(dst, "model-mtp.manifest.json")
        if keep:
            man = json.load(open(man_path))
            n_mtp_out = sum(len(v) for v in man["files"].values())
            assert acct["mtp_kept"] == n_mtp_out > 0, (acct, man)
        else:
            assert not os.path.exists(man_path)
        assert acct["vision_passthrough"] == n_vis, acct
        assert os.path.exists(os.path.join(dst, "model-mtp.safetensors")) == keep
        cfg = json.load(open(os.path.join(dst, "config.json")))
        assert cfg.get("text_config", {}).get("num_nextn_predict_layers", 3) == (3 if keep else 0)
        if not keep:
            assert cfg["num_nextn_predict_layers"] == 0
        model, args = load(dst)
        assert args.n_layers == CFG["n_layers"] and args.norm_eps == CFG["norm_eps"]
        assert len(model._vision_passthrough) == n_vis
        assert len(model._mtp_passthrough) == (n_mtp_out if keep else 0)
        model.set_token_map(list(range(VOCAB)))
        cache = model.make_cache(bsz=1, max_seq_len=64)
        ids = mx.array(rng.integers(3, VOCAB, size=(1, 20)))
        logits = model(ids, cache)
        lg = np.array(logits)
        assert np.isfinite(lg).all(), f"{tag}: non-finite logits"
        # decode continues from the loaded state
        logits2 = model(mx.array([[5]]), cache)
        assert np.isfinite(np.array(logits2)).all()
        if tag == "mxfp":
            q = cfg["quantization"]
            assert q["group_size"] == 32 and q["mode"] == "mxfp8", q
            assert q["modules"]["layers.0.attn.wq_b"] == \
                {"group_size": 32, "bits": 8, "mode": "mxfp8"}, q
            assert q["modules"]["layers.0.ffn.experts.gate_proj"] == \
                {"group_size": 32, "bits": 4, "mode": "mxfp4"}, q
            im = json.load(open(os.path.join(dst, "model.safetensors.index.json")))["weight_map"]
            assert "layers.0.attn.wq_b.scales" in im and "layers.0.attn.wq_b.biases" not in im
            assert "mtp.0.attn.wq_a.weight" in im
            assert "mtp.0.ffn.experts.gate_proj.weight" in im
        if tag == "omlx4":
            q = cfg["quantization"]
            assert "mode" not in q, q   # root mode would poison omlx's affine specs
            assert q["modules"]["layers.0.attn.wq_b"] == \
                {"group_size": 32, "bits": 4, "mode": "mxfp4"}, q
            im = json.load(open(os.path.join(dst, "model.safetensors.index.json")))["weight_map"]
            # experts stay individual (release-style names) for omlx's reader
            assert "layers.0.ffn.experts.0.w1.weight" in im, im
            shard = im["layers.0.attn.wq_b.weight"]
            tw = mx.load(os.path.join(dst, shard))["layers.0.attn.wq_b.weight"]
            assert tw.dtype == mx.uint8 and tw.shape[-1] % 4 == 0, tw
            # byte-packed storage must dequantize identically to mx-native words
            w = mx.load(os.path.join(dst, shard))
            d = mx.dequantize(w["layers.0.attn.wq_b.weight"].view(mx.uint32),
                              w["layers.0.attn.wq_b.scales"],
                              group_size=32, bits=4, mode="mxfp4")
            assert mx.isfinite(d).all().item()
        if tag == "omlx4" and importlib.util.find_spec("omlx"):
            # cross-check: omlx's source reader must digest the build end to end
            from omlx.patches.deepseek_v41.convert import convert as omlx_convert
            omlx_dst = os.path.join(root, "dst_omlx_converted")
            omlx_convert(dst, omlx_dst, preserve_mtp=True)
            ocfg = json.load(open(os.path.join(omlx_dst, "config.json")))
            specs = ocfg["omlx_deepseek_v41"]["quantized_modules"]
            assert specs and all(s["mode"] == "mxfp4" for s in specs.values()), \
                list(specs.items())[:2]
            shutil.rmtree(omlx_dst)
        print(f"  release round-trip [{tag}]: strict load ok, "
              f"{acct['kept']} tensors kept, "
              f"{acct['mtp_kept'] if keep else acct['dropped_mtp']} mtp "
              f"{'kept' if keep else 'dropped'}, "
              f"{n_vis} vision passthrough, forward+decode finite")

    # resume: a second convert over a finished build must skip every group
    # (manifests + shards present) and report identical accounting
    import time as _time
    t0 = _time.time()
    acct2 = convert(src, os.path.join(root, "dst_q8_4"),
                    bits=8, expert_bits=4, group_size=32, resume=True)
    assert acct2 == accts["q8_4"], (acct2, accts["q8_4"])
    print(f"  convert --resume over a finished build: all groups skipped "
          f"({_time.time() - t0:.2f}s)")

    # the omlx shim: header-only retag+split of a stored mxfp4 build must yield
    # the same tensors as --omlx-compat conversion of the same source
    from deepseek_v41_mlx.omlx_compat import shim_build
    dst_mxfp4 = os.path.join(root, "dst_plain_mxfp4")
    convert(src, dst_mxfp4, bits=parse_bits("mxfp4"), keep_mtp=False)
    shimmed = os.path.join(root, "dst_shimmed")
    shutil.copytree(dst_mxfp4, shimmed)
    shim_build(shimmed)

    def stacked(shard_path, view_bytes=False):
        out, parts = {}, {}
        for k, v in mx.load(shard_path).items():
            if view_bytes and v.dtype == mx.uint8 and k.endswith(".weight"):
                v = v.view(mx.uint32)
            m = re.match(r"((?:layers\.\d+)\.ffn\.experts)\.(\d+)\.(w[123])\.(weight|scales)$", k)
            if m:
                parts.setdefault((m.group(1), m.group(3), m.group(4)), {})[int(m.group(2))] = v
                continue
            out[k] = v
        for (base, wn, suf), ps in parts.items():
            proj = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[wn]
            out[f"{base}.{proj}.{suf}"] = mx.stack([ps[i] for i in sorted(ps)])
        return out

    shim_c = stacked(os.path.join(shimmed, "model-layers-00.safetensors"), view_bytes=True)
    compat_c = stacked(os.path.join(root, "dst_omlx4", "model-layers-00.safetensors"),
                       view_bytes=True)
    assert set(shim_c) == set(compat_c), (set(shim_c) ^ set(compat_c))
    for k in shim_c:
        assert str(shim_c[k].dtype) == str(compat_c[k].dtype) and \
            mx.array_equal(shim_c[k], compat_c[k]).item(), k
    model_s, _ = load(shimmed)
    print("  omlx shim: header-retagged build == --omlx-compat build, strict load ok")

    # strictness negative control: remove one tensor -> load must raise
    dst = os.path.join(root, "dst_bf16")
    broken = os.path.join(root, "dst_broken")
    shutil.copytree(dst, broken)
    smap = _json.load(open(os.path.join(broken, "model.safetensors.index.json")))["weight_map"]
    victim = "layers.4.attn.wkv.weight"
    shard = os.path.join(broken, smap[victim])
    w = dict(mx.load(shard))
    del w[victim]
    mx.save_safetensors(shard + ".new", w)   # write from the still-mapped source
    os.replace(shard + ".new.safetensors" if os.path.exists(shard + ".new.safetensors")
               else shard + ".new", shard)
    try:
        load(broken)
        raise AssertionError("strict load accepted a checkpoint with a missing tensor")
    except ValueError as e:
        assert "missing" in str(e)
    print("  strict-load negative control: missing tensor correctly refused")
    shutil.rmtree(root)


def stream_smoke():
    """stream.py: a layer-at-a-time pass over the raw tiny release must match
    the fully-loaded bf16 model (same math, same weights)."""
    import tempfile

    from deepseek_v41_mlx import stream as ST
    from deepseek_v41_mlx.convert import convert
    from deepseek_v41_mlx.layers import RMSNorm
    from deepseek_v41_mlx.load import load

    rng = np.random.default_rng(19)   # same seed -> same synthetic release
    root = tempfile.mkdtemp(prefix="dsv41_stream_")
    src = os.path.join(root, "src")
    _write_tiny_release(src, rng)
    dst = os.path.join(root, "dst")
    convert(src, dst)
    model, args = load(dst)
    model.set_token_map(list(range(VOCAB)))

    ids_np = np.random.default_rng(4).integers(3, VOCAB, size=(1, 20))
    cache = model.make_cache(bsz=1, max_seq_len=64)
    want = np.array(model(mx.array(ids_np), cache))

    raw_cfg, margs = ST.read_config(src)
    smap = ST.shard_map(src)
    top = ST.load_subset(src, smap, "embed.weight") | ST.load_subset(src, smap, "norm.weight") \
        | ST.load_subset(src, smap, "head.weight")
    state = ST.StreamState(margs, bsz=1, max_seq_len=64, token_map=list(range(VOCAB)))
    h = state.begin(ids_np, top["embed.weight"][mx.array(ids_np)], margs.hc_mult)
    for li in range(margs.n_layers):
        layer = ST.build_layer(margs, li, src, smap)
        h = ST.run_layer(layer, h, state)
        del layer
    norm = RMSNorm(margs.dim, margs.norm_eps)
    norm.weight = top["norm.weight"]
    got = np.array(ST.finish(h, state, norm, top["head.weight"], ids_np.shape[1]))
    d = rel_max_diff(got, want)
    assert d < 1e-5, f"streaming pass diverges from loaded model: {d}"
    import shutil
    shutil.rmtree(root)
    print(f"  stream.py layer-at-a-time pass vs loaded model: {d:.1e}")


def real_config_construction():
    """The real 40-layer config must construct without shape errors (lazy, no
    eval — the engram tables alone would be 197B parameters)."""
    cfg = json.load(open(os.path.join(ROOT, "docs", "reference", "hub_config.json")))
    args = ModelArgs.from_dict(cfg)
    for i in range(args.n_layers):
        if args.compress_ratio(i):
            args.kv_source_for(i)
            args.index_source_for(i)
    model = Model(args)
    from mlx.utils import tree_flatten
    n = len(tree_flatten(model.parameters()))
    total = sum(int(np.prod(v.shape)) for _, v in tree_flatten(model.parameters()))
    assert total > 7e11, total
    print(f"  real-config construction: {n} param tensors, {total/1e9:.1f}B params (lazy)")


def main():
    rng = np.random.default_rng(7)

    fake_quant_unit_parity()
    dequant_unit_parity()
    release_roundtrip()
    stream_smoke()
    real_config_construction()

    rargs = REF.ModelArgs(max_batch_size=B, max_seq_len=MAXSEQ, dtype="bf16",
                          expert_dtype=None, n_mtp_layers=0, vision_n_layers=0,
                          engram_pad_id=2, **CFG)
    margs = ModelArgs(max_seq_len=MAXSEQ, engram_pad_id=2, **CFG)

    with torch.no_grad():
        rm = REF.Transformer(rargs, tokenizer=None).float()
    mm = Model(margs, token_map=list(range(VOCAB)))
    transplant(rm, mm, rng)

    ids = rng.integers(3, VOCAB, size=(B, S + DECODE)).astype(np.int64)
    ids[:, 0] = 2           # engram pad token at the sequence start
    ids[0, 5] = 2           # and mid-sequence (an EOS/pad boundary inside the text)
    ids[1, 17] = 2

    results = {}

    # Pass 1 — QAT simulation OFF on both sides: pure continuous math, strict.
    set_fake_quant(False)
    battery(rm, mm, ids, results, "qat-off", 1e-4)

    # Pass 2 — QAT simulation ON (the real inference configuration). The
    # fake-quant ops themselves are bit-exact across implementations (unit test
    # above), but the grids amplify ~1e-7 upstream noise to a full code step
    # whenever a value lands on a rounding midpoint (verified: pre-quant values
    # differing by 1.2e-7 straddling the e4m3 midpoint -0.033203125 at layer 3,
    # batch 1, position 31), and one flipped window entry cascades down the
    # sequence tail through overlapping windows. All discrete decisions (gate
    # top-k, indexer sets, candidate blocks, compressed caches) still agree
    # exactly; this pass is held to 5e-3, with agreement ~4e-7 away from flips.
    set_fake_quant(True)
    r48, ref_dec, mfull, ref_full = battery(rm, mm, ids, results, "qat-on", 5e-3)

    # ---- the reference's stale index-key decode artifact, measured ----
    ref_prefill(rm, ids, S)                    # reset reference state to offset 48
    patch_indexer(False)
    raw_dec = []
    with torch.no_grad():
        for i in range(S, S + DECODE):
            _, lg, _ = rm(torch.from_numpy(ids[:, i:i + 1]), i)
            raw_dec.append(lg.float().numpy())
    patch_indexer(True)
    raw_dec = np.concatenate(raw_dec, axis=1)
    results["reference artifact: unpatched vs patched decode"] = rel_max_diff(raw_dec, ref_dec)

    # ---- negative controls: each break must move the logits ----
    def run_broken(setup, teardown):
        setup()
        try:
            out, _ = ours_prefill(mm, ids, S)
        finally:
            teardown()
        return rel_max_diff(out, r48)

    d = run_broken(
        lambda: [setattr(l.attn, "_break_rope_inverse", True) for l in mm.layers],
        lambda: [setattr(l.attn, "_break_rope_inverse", False) for l in mm.layers])
    results["NEGATIVE rope-inverse broken"] = d
    assert d > 1e-3, f"breaking rope-inverse barely moved logits ({d}) — dead path?"

    d = run_broken(
        lambda: [setattr(l.attn, "_break_sink", True) for l in mm.layers],
        lambda: [setattr(l.attn, "_break_sink", False) for l in mm.layers])
    results["NEGATIVE attention sinks zeroed"] = d
    assert d > 1e-3, f"zeroing sinks barely moved logits ({d})"

    d = run_broken(lambda: setattr(mm, "_break_sharing", True),
                   lambda: setattr(mm, "_break_sharing", False))
    results["NEGATIVE cross-layer sharing severed"] = d
    assert d > 1e-3, f"severing sharing barely moved logits ({d})"

    # sanity after teardown: the model still matches
    m_again, _ = ours_prefill(mm, ids, S)
    assert rel_max_diff(m_again, r48) < 5e-3

    print()
    for k, v in results.items():
        print(f"  {k:55s} {v:.3e}")
    corr = np.corrcoef(mfull.ravel(), ref_full.ravel())[0, 1]
    am = (mfull.argmax(-1) == ref_full.argmax(-1)).mean()
    print(f"  {'logit correlation (56-token prefill, all positions)':55s} {corr:.8f}")
    print(f"  {'argmax agreement':55s} {am * 100:.1f}%")
    assert am == 1.0
    print("\nPARITY PASS")
    return results


if __name__ == "__main__":
    main()
