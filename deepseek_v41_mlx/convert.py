"""Convert the raw DeepSeek-V4.1-Flash release to an MLX build.

The hub release already uses inference-style names (``layers.N.attn.wq_a.weight``,
``ffn``, ``.scale`` — no ``model.`` prefix, no ``self_attn``/``mlp``/
``weight_scale_inv``), so sanitizing is mostly dequantization plus expert
stacking:

* fp8 ``weight``+``scale`` pairs (32x32 ue8m0 blocks) -> bf16;
* fp4 routed experts (packed nibbles, per-32 ue8m0) -> bf16, stacked per layer
  into SwitchGLU's ``experts.{gate,up,down}_proj.weight`` (w1/w3/w2);
* ``engram.embed`` stays in its native fp8+scale form (dequantizing ~197B
  parameters to bf16 would double 40% of the model); both tensors pass through
  as uint8;
* ``hc_*``, ``attn_sink``, ``gate.bias``, ``gate.bias_vl`` -> fp32;
* ``mtp.*`` (the DSpark/MTP draft stack) is kept by default — sanitized and
  quantized like the backbone (its routed experts are fp4 too) and written to
  its own ``model-mtp`` shard; the tied ``mtp.N.embed``/``mtp.N.head`` clones
  are dropped (the reference ties them to the backbone's tensors). The
  text-only runtime carries the retained tensors as declared passthrough.
  ``keep_mtp=False`` drops the stack (earlier builds did);
* the vision tower (``vision.*``, ``aligner.*``, ``image_start/end/newline``)
  is carried through **unchanged** under its original names. The text-only
  runtime does not load it, but the tensors stay in the build (they are bf16
  and small, ~0.8 GB) so nothing silently vanishes; ``load.py`` accounts for
  them explicitly.

Quantization (optional): applied per module, experts at ``expert_bits`` and
everything else at ``bits`` — each an affine bit width (2-8) or a microscaling
format, ``{"bits": 4, "mode": "mxfp4"}`` as :func:`parse_bits` returns:
E2M1/E4M3 elements sharing one e8m0 scale per group of 32 and no bias term, so
the group size is pinned to 32 for those modules. ``wo_a`` is NEVER quantized:
attention reshapes its weight to ``[o_groups, o_lora_rank, -1]`` and a
quantized Linear's ``.weight`` is packed uint32 — the reshape would carve
packed bytes, not logical weights. The same predicate is replayed at load time.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import mlx.core as mx

from .config import ModelArgs
from .dequant import dequant_fp4, dequant_fp8, is_fp4_expert

VISION_PREFIXES = ("vision.", "aligner.", "image_start", "image_end", "image_newline")

_MXFP_RE = re.compile(r"^mx(?:f(?:p)?)?([48])$")


def parse_bits(value) -> dict:
    """A --bits value -> {"bits", "mode"}: affine widths (2-8) or mxfp4/mxfp8
    (mxf4/mxf8 accepted)."""
    text = str(value).strip().lower()
    if m := _MXFP_RE.match(text):
        return {"bits": int(m.group(1)), "mode": "mxfp" + m.group(1)}
    try:
        return {"bits": int(text), "mode": "affine"}
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid bits {value!r}: use 2-8, mxfp4 or mxfp8") from None


def quant_params(mode_bits: dict, group_size: int) -> dict:
    """Complete per-module params; the mx spec requires (and this pins) a group
    size of 32 for the microscaling modes."""
    mode = mode_bits["mode"]
    out = {"group_size": 32 if mode != "affine" else group_size, "bits": mode_bits["bits"]}
    if mode != "affine":
        out["mode"] = mode
    return out


def fmt_bits(params: dict) -> str:
    mode = params.get("mode", "affine")
    return mode if mode != "affine" else f"{params['bits']}b"


def _family(value, group_size: int) -> dict | None:
    """int | {"bits", "mode"} | None -> complete params dict | None."""
    if value is None:
        return None
    if isinstance(value, dict):
        return quant_params(value, group_size)
    return quant_params({"bits": int(value), "mode": "affine"}, group_size)

# modules kept at bf16 even when quantizing (beyond shape-ineligible ones):
#   wo_a          — block-diagonal reshape carves packed bytes (see module doc)
#   engram.embed  — stays in native fp8+ue8m0, dequantized per lookup
#   compressor.*  — tiny, and the ratio>1 projections run in fp32
#   indexer.wk / weights_proj — tiny, and they drive discrete top-k selection
_NEVER_QUANT = re.compile(
    r"(\.wo_a\.|\.engram\.embed\.|\.compressor\.|\.indexer\.wk\.|\.weights_proj\.)")


def is_quant_target(name: str, in_dim: int, group_size: int) -> bool:
    if not name.endswith(".weight"):
        return False
    if _NEVER_QUANT.search(name):
        return False
    if in_dim % group_size:
        return False
    tail = name.rsplit(".", 2)
    quantizable = ("wq_a", "wq_b", "wkv", "wo_b", "w1", "w2", "w3",
                   "gate_proj", "up_proj", "down_proj", "embed", "head")
    return len(tail) >= 2 and tail[-2] in quantizable


def bits_for(name: str, bits, expert_bits=None, group_size: int = 64) -> dict | None:
    """Complete params for one module: routed experts at ``expert_bits`` when
    given, everything else at ``bits``. Either argument takes an affine width
    or a ``{"bits", "mode"}`` microscaling spec; ``None`` means no quantization."""
    if expert_bits is not None and is_fp4_expert(name):
        return _family(expert_bits, group_size)
    return _family(bits, group_size)


def _shard_iter(src: str):
    index_path = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(index_path):
        wmap = json.load(open(index_path))["weight_map"]
        return wmap
    wmap = {}
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        for k in mx.load(f):
            wmap[k] = os.path.basename(f)
    return wmap


def _to_u8(a: mx.array) -> mx.array:
    """Raw release bytes (fp8 values / ue8m0 scales) as uint8, however stored."""
    if a.dtype in (mx.uint8, mx.int8):
        return a.view(mx.uint8)
    return a.view(mx.uint8)


def sanitize_group(names: list[str], tensors: dict[str, mx.array],
                   dtype=mx.bfloat16, stack_experts: bool = True) -> dict[str, mx.array]:
    """Sanitize one coherent group of raw tensors (a layer, the MTP stack, or
    the top level).

    Returns final-name -> array. ``mtp.*`` sanitizes exactly like a backbone
    layer (retention is the caller's choice); the tied ``mtp.N.embed`` /
    ``mtp.N.head`` clones are dropped; vision tensors pass through unchanged.

    With ``stack_experts=False`` (omlx compat) routed experts keep their
    release-style individual names — ``...ffn.experts.{i}.{w1,w3,w2}.weight`` —
    which omlx's source reader requires (it stacks them itself); the caller's
    loader re-stacks them into the runtime's SwitchGLU layout.
    """
    out: dict[str, mx.array] = {}
    consumed = set()
    expert_parts: dict[str, dict[int, mx.array]] = {}

    for name in names:
        if name in consumed:
            continue
        if name.startswith(VISION_PREFIXES):
            out[name] = tensors[name]
            continue
        if re.match(r"mtp\.\d+\.(embed|head)\.weight$", name):
            # tied to the backbone's embed/head — the reference drops the clones
            scale_name = name[:-len(".weight")] + ".scale"
            if scale_name in tensors:
                consumed.add(scale_name)
            continue
        if name.endswith(".scale"):
            continue  # handled with its weight
        t = tensors[name]

        if ".engram.embed." in name:  # stays native fp8 + ue8m0
            out[name] = _to_u8(t)
            scale_name = name[:-len(".weight")] + ".scale"
            out[scale_name] = _to_u8(tensors[scale_name])
            consumed.add(scale_name)
            continue

        scale_name = name[:-len(".weight")] + ".scale" if name.endswith(".weight") else None
        has_scale = scale_name in tensors if scale_name else False
        if has_scale:
            consumed.add(scale_name)
            s = _to_u8(tensors[scale_name])
            if is_fp4_expert(name):
                w = dequant_fp4(t, s, dtype)
            else:
                w = dequant_fp8(_to_u8(t), s, dtype)
        else:
            w = t

        m = re.match(r"((?:layers\.\d+|mtp\.\d+)\.ffn\.experts)\.(\d+)\.(w[123])\.weight$", name)
        if m and stack_experts:
            prefix, idx, wn = m.group(1), int(m.group(2)), m.group(3)
            proj = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[wn]
            expert_parts.setdefault(f"{prefix}.{proj}.weight", {})[idx] = w
            continue

        if re.search(r"(hc_\w+|attn_sink|gate\.bias(_vl)?)$", name):
            w = w.astype(mx.float32)
        out[name] = w

    for final, parts in expert_parts.items():
        n = len(parts)
        assert sorted(parts) == list(range(n)), f"missing experts for {final}"
        out[final] = mx.stack([parts[i] for i in range(n)])

    orphans = [n for n in names
               if n.endswith(".scale") and n not in consumed
               and not n.startswith(VISION_PREFIXES)]
    if orphans:
        raise ValueError(f"{len(orphans)} scale tensors without a weight, "
                         f"e.g. {orphans[:3]} — layout change?")
    return out


def quantize_tree(weights: dict[str, mx.array], bits, expert_bits,
                  group_size: int = 64, omlx_compat: bool = False) -> tuple[dict[str, mx.array], dict[str, dict]]:
    """Replace eligible ``.weight`` tensors with MLX quantized tuples, matching
    what ``nn.quantize`` produces at load time: affine (weight, scales, biases)
    triples, or (weight, scales) pairs for mxfp4/mxfp8 modules — no bias term,
    group size pinned to 32. ``bits``/``expert_bits`` take an affine width or a
    ``{"bits", "mode"}`` microscaling spec.

    With ``omlx_compat`` (omlx's deepseek_v41 source reader): mxfp4 weights are
    stored byte-packed — the same words viewed as uint8, retagged U8 + F8_E8M0
    scales at save time, which omlx's native fp4 branch consumes losslessly;
    mxfp8 has no lossless source path there (per-row e8m0 scales vs the official
    32-row-block layout), so those modules fall back to affine 8-bit.

    Also returns the per-module map {module path: {group_size, bits[, mode]}}
    that config.json carries for the loader to replay."""
    out = {}
    modules: dict[str, dict] = {}
    for name, w in weights.items():
        if name.startswith(VISION_PREFIXES):
            out[name] = w
            continue
        p = bits_for(name, bits, expert_bits, group_size)
        if p is None or not is_quant_target(name, w.shape[-1], p["group_size"]):
            out[name] = w
            continue
        mode = p.get("mode", "affine")
        if omlx_compat and mode == "mxfp8":
            g = group_size if w.shape[-1] % group_size == 0 else p["group_size"]
            p = {"group_size": g, "bits": 8}
            mode = "affine"
        q = mx.quantize(w, group_size=p["group_size"], bits=p["bits"], mode=mode)
        weight = q[0]
        if omlx_compat and mode == "mxfp4":
            weight = weight.view(mx.uint8)   # byte-packed nibbles; identical words
        base = name[:-len(".weight")]
        out[f"{base}.weight"] = weight
        out[f"{base}.scales"] = q[1]
        if len(q) == 3:
            out[f"{base}.biases"] = q[2]
        entry = {"group_size": p["group_size"], "bits": p["bits"]}
        if mode != "affine":
            entry["mode"] = mode
        if omlx_compat:
            # individual experts share the family spec of their stacked runtime module
            m = re.match(r"((?:layers\.\d+|mtp\.\d+)\.ffn\.experts)\.\d+\.(w[123])$", base)
            if m:
                proj = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[m.group(2)]
                modules.setdefault(f"{m.group(1)}.{proj}", dict(entry))
                continue
        modules[base] = entry
    return out, modules


def quantize_engram_table(weight_u8: mx.array, scale_u8: mx.array, bits,
                          group_size: int, step: int = 1 << 22) -> dict[str, mx.array]:
    """Native fp8+ue8m0 table (already in memory) -> MLX quantized tuple, in
    row chunks (affine: weight/scales/biases; mxfp: weight/scales). NOTE:
    evaluating any chunk of a lazily-loaded mx tensor materializes the WHOLE
    load node, so for the 98 GB release tables use
    :func:`quantize_engram_table_from_file`, which slices rows from disk.
    This in-memory variant serves tests and the ladder's gathered rows."""
    from .dequant import dequant_fp8_rows
    p = _family(bits, group_size)
    kw = dict(group_size=p["group_size"], bits=p["bits"], mode=p.get("mode", "affine"))
    rows = weight_u8.shape[0]
    cw, cs, cb = [], [], []
    for i in range(0, rows, step):
        deq = dequant_fp8_rows(weight_u8[i:i + step], scale_u8[i:i + step])
        q = mx.quantize(deq.astype(mx.bfloat16), **kw)
        mx.eval(q)
        cw.append(q[0]); cs.append(q[1])
        if len(q) == 3:
            cb.append(q[2])
    out = {"weight": mx.concatenate(cw), "scales": mx.concatenate(cs)}
    if cb:
        out["biases"] = mx.concatenate(cb)
    return out


def quantize_engram_table_from_file(src_dir: str, wmap: dict, wname: str,
                                    bits, group_size: int,
                                    step: int = 1 << 22) -> dict[str, mx.array]:
    """Stream a release engram table off disk in row chunks (safetensors row
    slicing — never materializes the 98 GB tensor) into an MLX quantized tuple
    (affine: weight/scales/biases; mxfp: weight/scales). Chunk arithmetic is
    identical to :func:`quantize_engram_table` (quantization is row-local)."""
    import torch
    from safetensors import safe_open
    from .dequant import dequant_fp8_rows

    p = _family(bits, group_size)
    kw = dict(group_size=p["group_size"], bits=p["bits"], mode=p.get("mode", "affine"))
    sname = wname[:-len(".weight")] + ".scale"
    fw = safe_open(os.path.join(src_dir, wmap[wname]), framework="pt")
    fs = fw if wmap[sname] == wmap[wname] else \
        safe_open(os.path.join(src_dir, wmap[sname]), framework="pt")
    slw, sls = fw.get_slice(wname), fs.get_slice(sname)
    rows = slw.get_shape()[0]
    cw, cs, cb = [], [], []
    for i in range(0, rows, step):
        w_u8 = mx.array(slw[i:i + step].view(torch.uint8).numpy())
        s_u8 = mx.array(sls[i:i + step].view(torch.uint8).numpy())
        deq = dequant_fp8_rows(w_u8, s_u8)
        q = mx.quantize(deq.astype(mx.bfloat16), **kw)
        mx.eval(q)
        cw.append(q[0]); cs.append(q[1])
        if len(q) == 3:
            cb.append(q[2])
        del w_u8, s_u8, deq
        mx.clear_cache()
    out = {"weight": mx.concatenate(cw), "scales": mx.concatenate(cs)}
    if cb:
        out["biases"] = mx.concatenate(cb)
    return out


def _group_tag(g: str) -> str:
    if g.startswith("layers."):
        return f"layers-{int(g.split('.')[1]):02d}"
    return g


def _stream_raw_copy(src_dir: str, wmap: dict, names: list[str], out_path: str,
                     chunk_bytes: int = 1 << 28):
    """Byte-stream tensors from source shards into one new safetensors file,
    preserving their stored dtype tags, without materializing them (the native
    engram tables are ~101 GB each). All passthrough dtypes are 1 byte/elt."""
    import struct
    from safetensors import safe_open

    handles: dict[str, object] = {}
    entries = []
    off = 0
    for n in names:
        f = os.path.join(src_dir, wmap[n])
        if f not in handles:
            handles[f] = safe_open(f, framework="pt")
        sl = handles[f].get_slice(n)
        shape, dt = sl.get_shape(), sl.get_dtype()
        nbytes = 1
        for s in shape:
            nbytes *= s
        entries.append((n, dt, shape, off, off + nbytes, sl))
        off += nbytes
    header = {n: {"dtype": dt, "shape": list(shape), "data_offsets": [a, b]}
              for n, dt, shape, a, b, _ in entries}
    hjson = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(hjson)) % 8
    hjson += b" " * pad
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(hjson)))
        out.write(hjson)
        for n, dt, shape, a, b, sl in entries:
            row_bytes = max(1, (b - a) // max(shape[0], 1))
            step = max(1, chunk_bytes // row_bytes)
            for i in range(0, shape[0], step):
                t = sl[i:i + step]
                out.write(t.view(__import__("torch").uint8).numpy().tobytes())
    os.replace(tmp, out_path)


def _read_st_header(path: str) -> dict:
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def convert(src: str, dst: str, bits: int | dict | None = None,
            expert_bits: int | dict | None = None, engram_bits: int | dict | None = None,
            group_size: int = 64, dtype=mx.bfloat16, resume: bool = False,
            keep_mtp: bool = True, omlx_compat: bool = False) -> dict:
    """Stream the release group by group (top level, each layer, MTP, vision)
    into an MLX build at ``dst`` — one output shard per group, written to a
    temp name and renamed on completion, so ``resume=True`` skips every
    finished group at a clean layer boundary.

    ``bits``/``expert_bits``/``engram_bits`` take an affine width (2-8) or a
    microscaling spec ``{"bits": 4, "mode": "mxfp4"}`` (see :func:`parse_bits`).

    ``engram_bits`` re-quantizes the two engram tables from native fp8+ue8m0
    (203 GB total), streamed in row chunks (never materialized whole);
    ``None`` keeps them native, byte-streamed into their own shard.

    ``keep_mtp=False`` drops the ``mtp.*`` draft stack (earlier builds did);
    the default retains it — sanitized and quantized with the rest, written
    to its own ``model-mtp`` shard, carried as declared passthrough by the
    loader.

    ``omlx_compat`` stores mxfp4 weights byte-packed (U8 + F8_E8M0 scales,
    consumed losslessly by omlx's native fp4 branch) and downgrades mxfp8
    modules to affine 8-bit (omlx's source reader has no mxfp8 path); the
    root quantization block carries no ``mode`` omlx could mis-merge.

    ``config.json`` gets a ``quantization`` block with defaults plus an
    explicit per-module map keyed by runtime-internal module paths.
    Returns {kept, dropped_mtp, mtp_kept, vision_passthrough}.
    """
    os.makedirs(dst, exist_ok=True)
    wmap = _shard_iter(src)

    def owner(name: str) -> str:
        m = re.match(r"(layers\.\d+)\.", name)
        if m:
            return m.group(1)
        if name.startswith("mtp."):
            return "mtp"
        if name.startswith(VISION_PREFIXES):
            return "vision"
        return "top"

    groups: dict[str, list[str]] = {}
    for name in wmap:
        groups.setdefault(owner(name), []).append(name)
    mtp_names = groups.pop("mtp", [])
    accounting = {"kept": 0,
                  "dropped_mtp": len(mtp_names) if not keep_mtp else 0,
                  "mtp_kept": 0,
                  "vision_passthrough": 0}

    shard_cache: dict[str, dict] = {}

    def fetch(names):
        res = {}
        for n in names:
            f = wmap[n]
            if f not in shard_cache:
                shard_cache.clear()
                shard_cache[f] = mx.load(os.path.join(src, f))
            res[n] = shard_cache[f][n]
        return res

    if not wmap:
        raise FileNotFoundError(
            f"no safetensors found under {src!r} — check the snapshot path "
            f"(model.safetensors.index.json or *.safetensors expected)")
    order = (["top"] if "top" in groups else []) + \
        sorted((g for g in groups if g.startswith("layers.")),
               key=lambda s: int(s.split(".")[1]))
    if keep_mtp and mtp_names:
        order.append("mtp")
    if "vision" in groups:
        order.append("vision")

    index: dict[str, str] = {}
    module_map: dict[str, dict] = {}
    t0 = time.time()
    engram_eff = engram_bits
    if omlx_compat and isinstance(engram_bits, dict) and engram_bits.get("mode", "affine") != "affine":
        engram_eff = {"bits": engram_bits["bits"], "mode": "affine"}
        print(f"[omlx-compat] engram tables: mxfp{engram_bits['bits']} -> affine "
              f"{engram_bits['bits']}b (no mxfp engram path in omlx's source reader)",
              flush=True)

    for gi, g in enumerate(order, 1):
        tag = _group_tag(g)
        names = sorted(mtp_names if g == "mtp" else groups[g])
        out_files = {"main": os.path.join(dst, f"model-{tag}.safetensors")}
        engram_pair = [n for n in names if ".engram.embed." in n]
        if engram_pair:
            out_files["engram"] = os.path.join(dst, f"model-{tag}-engram.safetensors")
        manifest_path = os.path.join(dst, f"model-{tag}.manifest.json")

        if resume and os.path.exists(manifest_path) and \
                all(os.path.exists(p) for p in out_files.values()):
            man = json.load(open(manifest_path))
            cached = True
        else:
            cached = False
            print(f"[{gi}/{len(order)}] {g}: {len(names)} tensors — "
                  f"sanitizing + quantizing...", flush=True)
            work = [n for n in names if n not in engram_pair]
            sane = sanitize_group(work, fetch(work), dtype,
                                  stack_experts=not omlx_compat)
            grp_modules: dict[str, dict] = {}
            if bits and g != "vision":
                sane, grp_modules = quantize_tree(sane, bits, expert_bits, group_size,
                                                  omlx_compat=omlx_compat)
            for v in sane.values():
                mx.eval(v)
            tmp = out_files["main"] + ".tmp.safetensors"
            mx.save_safetensors(tmp, sane)
            os.replace(tmp, out_files["main"])
            if omlx_compat and grp_modules:
                from .omlx_compat import mxfp4_dtypes, retag_shard
                dtypes = mxfp4_dtypes(grp_modules)
                if dtypes:
                    # expand stacked expert entries to the individual names
                    expanded = {}
                    for key, dt in dtypes.items():
                        m = re.match(r"((?:layers\.\d+|mtp\.\d+)\.ffn\.experts)\."
                                     r"(gate_proj|up_proj|down_proj)\.(weight|scales)$", key)
                        if m:
                            wn = {"gate_proj": "w1", "up_proj": "w3",
                                  "down_proj": "w2"}[m.group(2)]
                            pat = re.compile(rf"{re.escape(m.group(1))}\.\d+\.{wn}\.{m.group(3)}$")
                            expanded.update({k: dt for k in sane if pat.match(k)})
                        else:
                            expanded[key] = dt
                    retag_shard(out_files["main"], expanded)
            engram_names = []
            if engram_pair and engram_bits:
                print(f"  engram table -> re-quantizing "
                      f"({fmt_bits(_family(engram_eff, group_size))}) in row chunks...",
                      flush=True)
                wname = next(n for n in engram_pair if n.endswith(".weight"))
                base = wname[: -len(".weight")]
                triple = quantize_engram_table_from_file(src, wmap, wname,
                                                         engram_eff, group_size)
                shard = {f"{base}.{suf}": v for suf, v in triple.items()}
                tmp = out_files["engram"] + ".tmp.safetensors"
                mx.save_safetensors(tmp, shard)
                os.replace(tmp, out_files["engram"])
                engram_names = sorted(shard)
                del triple, shard
                mx.clear_cache()
            elif engram_pair:
                print("  engram table -> streaming native fp8+ue8m0 passthrough...",
                      flush=True)
                _stream_raw_copy(src, wmap, engram_pair, out_files["engram"])
                engram_names = engram_pair
            man = {"files": {os.path.basename(out_files["main"]): sorted(sane),
                             **({os.path.basename(out_files["engram"]): engram_names}
                                if engram_pair else {})},
                   "modules": grp_modules}
            json.dump(man, open(manifest_path, "w"))
            del sane
            mx.clear_cache()

        for fname, tnames in man["files"].items():
            for t in tnames:
                index[t] = fname
            accounting["kept"] += len(tnames)
            if g == "vision":
                accounting["vision_passthrough"] += len(tnames)
            elif g == "mtp":
                accounting["mtp_kept"] += len(tnames)
        module_map.update(man["modules"])
        size = sum(os.path.getsize(p) for p in out_files.values())
        print(f"[{gi}/{len(order)}] {g}: {sum(len(v) for v in man['files'].values())} tensors out, "
              f"{len(man['modules'])} modules quantized, {size / 1e9:.2f} GB"
              f"{' (cached)' if cached else ''} — {time.time() - t0:.0f}s elapsed", flush=True)

    total_gb = sum(os.path.getsize(os.path.join(dst, f)) for f in set(index.values()))
    print(f"done: {len(index)} tensors in {len(set(index.values()))} shards, "
          f"{total_gb / 1e9:.1f} GB — {time.time() - t0:.0f}s total", flush=True)

    if omlx_compat and engram_bits:
        # omlx resolves quantized engram tables through the module map too
        for k in index:
            if k.endswith(".engram.embed.scales"):
                module_map.setdefault(k[:-len(".scales")],
                                      {"group_size": group_size,
                                       "bits": _family(engram_eff, group_size)["bits"]})
    if omlx_compat:
        # individual affine experts need their own spec entries for omlx's reader
        # (mxfp4 experts ride the dtype-tagged native branch and need none)
        root_fam = _family(bits, group_size) or {}
        for key, entry in list(module_map.items()):
            m = re.match(r"((?:layers\.\d+)\.ffn\.experts)\.(gate_proj|up_proj|down_proj)$", key)
            if not m or entry.get("mode", "affine") != "affine" or entry == root_fam:
                continue
            wn = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}[m.group(2)]
            pat = re.compile(rf"{re.escape(m.group(1))}\.(\d+)\.{wn}\.weight$")
            for k in index:
                mm = pat.match(k)
                if mm:
                    module_map.setdefault(f"{m.group(1)}.{mm.group(1)}.{wn}", dict(entry))

    json.dump({"metadata": {}, "weight_map": index},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"))

    cfg_path = os.path.join(src, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    out_cfg = dict(cfg)
    out_cfg["model_type"] = "deepseek_v41"
    out_cfg.pop("quantization_config", None)   # describes the raw release, not this build
    if not keep_mtp:
        out_cfg["num_nextn_predict_layers"] = 0
        if "text_config" in out_cfg:
            out_cfg["text_config"]["num_nextn_predict_layers"] = 0
    if bits or engram_bits:
        if omlx_compat:
            # no root "mode" (omlx merges root defaults into every affine spec)
            # and flat per-module entries (omlx reads mlx_lm-style overrides)
            root_bits = bits["bits"] if isinstance(bits, dict) else bits
            q = {"group_size": group_size, "bits": root_bits}
            for path, entry in module_map.items():
                q[path] = dict(entry)
        else:
            q = dict(_family(bits, group_size) or {"group_size": group_size})
        q["expert_bits"] = _family(expert_bits, group_size)
        q["engram_bits"] = _family(engram_eff, group_size)
        q["modules"] = module_map
        out_cfg["quantization"] = q
        out_cfg["quantization_config"] = q
    json.dump(out_cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)

    for f in ("tokenizer.json", "tokenizer_config.json"):
        p = os.path.join(src, f)
        if os.path.exists(p):
            import shutil
            shutil.copyfile(p, os.path.join(dst, f))
    return accounting
