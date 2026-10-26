"""omlx compatibility for stored mxfp builds.

omlx's ``deepseek_v41`` source reader dispatches on safetensors dtype tags:

* native MXFP4 = byte-packed (``U8``) weights + ``F8_E8M0`` scales — consumed
  by its fp4 branch with no quantization config at all;
* affine = ``U32`` weights + float scales **and** biases, resolved from the
  config's per-module map (root defaults merge into every spec, so a root
  ``mode`` poisons affine modules);
* experts must be **individual** release-style tensors
  (``...experts.{i}.{w1,w3,w2}.weight``, rank two) — omlx stacks them itself;
  mx-style mxfp packing (``U32`` weights, ``U8``-tagged scales, experts stacked
  3-D) matches none of that, and mxfp8 has no lossless source path (per-row
  e8m0 scales vs the official 32-row-block layout).

``mx.quantize(mxfp4)`` packs exactly the bytes the fp4 branch expects — only
the tags and the expert stacking differ, and both are header arithmetic.
:func:`retag_shard` rewrites tags (and optionally splits stacked experts)
without touching tensor bytes; :func:`shim_build` applies that to a whole
stored build, in place.
"""

from __future__ import annotations

import json
import os
import re
import struct

_WN_OF = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


def mxfp4_dtypes(modules: dict) -> dict[str, str]:
    """{tensor key: new dtype tag} for every mxfp4 module in a quantization map."""
    out = {}
    for mod, entry in (modules or {}).items():
        if isinstance(entry, dict) and entry.get("mode") == "mxfp4":
            out[mod + ".weight"] = "U8"
            out[mod + ".scales"] = "F8_E8M0"
    return out


def mxfp8_modules(modules: dict) -> list[str]:
    """Modules whose mode cannot be expressed for omlx's source reader."""
    return [mod for mod, entry in (modules or {}).items()
            if isinstance(entry, dict) and entry.get("mode") == "mxfp8"]


def _rewrite_header(path: str, header: dict, backup_raw: bytes | None) -> None:
    """Replace a shard's header in place, keeping tensor bytes and the 8-byte
    alignment. data_offsets are header-relative, so entries may be retagged,
    added, or removed freely. When the new header is longer, the data region
    shifts right by the (few-byte) delta in one backward pass."""
    body = json.dumps(header, separators=(",", ":")).encode()
    body += b" " * ((-len(body)) % 8)
    old_len = struct.unpack("<Q", backup_raw[:8])[0] if backup_raw is not None else None
    with open(path, "r+b", buffering=0) as f:
        if old_len is None:
            old_len = struct.unpack("<Q", f.read(8))[0]
        if len(body) <= old_len:
            body += b" " * (old_len - len(body))
            f.seek(0)
            f.write(struct.pack("<Q", len(body)))
            f.write(body)
            return
        delta = len(body) - old_len
        size = os.path.getsize(path)
        f.truncate(size + delta)
        data_start = 8 + old_len
        chunk = 1 << 26
        buf = bytearray(chunk)
        pos = size
        while pos > data_start:
            step = min(chunk, pos - data_start)
            f.seek(pos - step)
            view = memoryview(buf)[:step]
            f.readinto(view)
            f.seek(pos - step + delta)
            f.write(view)
            pos -= step
        f.seek(0)
        f.write(struct.pack("<Q", len(body)))
        f.write(body)


def _read_header(path: str) -> tuple[dict, bytes]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    return json.loads(raw), struct.pack("<Q", n) + raw


def _split_stacked_experts(header: dict, dtypes: dict[str, str]) -> dict[str, int]:
    """Turn stacked expert tensors (``...experts.{proj}.{suf}``) into omlx's
    individual ``...experts.{i}.{w1,w3,w2}.{suf}`` entries: expert i is a
    contiguous slab of the stacked tensor, so this is header arithmetic only.
    The expert count comes from each tensor's own leading dim. Returns
    {stacked key: expert count} for the entries that were split."""
    split = {}
    for key, dtype in list(dtypes.items()):
        m = re.match(r"((?:layers\.\d+|mtp\.\d+)\.ffn\.experts)\."
                     r"(gate_proj|up_proj|down_proj)\.(weight|scales)$", key)
        if not m or key not in header:
            continue
        entry = header.pop(key)
        shape = entry["shape"]
        n = shape[0]
        split[key] = n
        start, end = entry["data_offsets"]
        step = (end - start) // n
        assert step * n == end - start, key
        individual_shape = shape[1:]
        if entry["dtype"] == "U32" and dtype == "U8":
            individual_shape = individual_shape[:-1] + [individual_shape[-1] * 4]
        base, proj, suf = m.groups()
        for i in range(n):
            header[f"{base}.{i}.{_WN_OF[proj]}.{suf}"] = {
                "data_offsets": [start + i * step, start + (i + 1) * step],
                "dtype": dtype,
                "shape": individual_shape,
            }
    return split


def retag_shard(path: str, dtypes: dict[str, str], backup: bool = False) -> bool:
    """Rewrite dtype/shape tags of one safetensors shard in place (tensor bytes
    untouched). Returns True when anything changed."""
    header, raw = _read_header(path)
    touched = False
    for key, dtype in dtypes.items():
        entry = header.get(key)
        if entry is None or entry["dtype"] == dtype:
            continue
        if entry["dtype"] == "U32" and dtype == "U8":
            entry["shape"] = entry["shape"][:-1] + [entry["shape"][-1] * 4]
        entry["dtype"] = dtype
        touched = True
    if not touched:
        return False
    if backup:
        with open(path + ".header.bak", "wb") as f:
            f.write(raw)
    _rewrite_header(path, header, raw if not backup else None)
    return True


def shim_build(build_dir: str, dry_run: bool = False) -> dict:
    """Make a stored mxfp4 build omlx-loadable, in place.

    Retags and splits every mxfp4 module per :func:`retag_shard`, rewrites the
    weight map to individual expert keys, and drops the root quantization
    ``mode`` from config.json. Refuses builds with mxfp8 modules (no lossless
    omlx source path — re-convert with ``--omlx-compat``, which stores them as
    affine 8-bit). Returns counts.
    """
    build_dir = os.path.abspath(build_dir)
    cfg_path = os.path.join(build_dir, "config.json")
    cfg = json.load(open(cfg_path))
    quant = cfg.get("quantization") or {}
    modules = quant.get("modules") or {}
    blocked = mxfp8_modules(modules)
    if blocked:
        raise SystemExit(
            f"{len(blocked)} mxfp8 modules cannot be shimmed (e.g. {blocked[:3]}) — "
            f"re-convert with --omlx-compat instead")
    dtypes = mxfp4_dtypes(modules)
    if not dtypes:
        print("no mxfp4 modules found — nothing to do")
        return {"shards": 0, "modules": 0}

    index_path = os.path.join(build_dir, "model.safetensors.index.json")
    index = json.load(open(index_path))
    weight_map = index["weight_map"]
    shards = sorted({f for k, f in weight_map.items()
                     if k in dtypes or k.endswith(".scales")})
    n_experts = (cfg.get("text_config") or cfg).get("n_routed_experts")
    print(f"{len(dtypes) // 2} mxfp4 modules in {len(shards)} shards "
          f"({n_experts} routed experts)")
    if dry_run:
        return {"shards": len(shards), "modules": len(dtypes) // 2}

    all_splits: dict[str, int] = {}
    for i, shard in enumerate(shards, 1):
        path = os.path.join(build_dir, shard)
        header, raw = _read_header(path)
        touched = False
        for key, dtype in dtypes.items():
            entry = header.get(key)
            if entry is None or entry["dtype"] == dtype:
                continue
            if entry["dtype"] == "U32" and dtype == "U8":
                entry["shape"] = entry["shape"][:-1] + [entry["shape"][-1] * 4]
            entry["dtype"] = dtype
            touched = True
        split = _split_stacked_experts(header, dtypes)
        all_splits.update(split)
        if touched or split:
            with open(path + ".header.bak", "wb") as f:
                f.write(raw)
            _rewrite_header(path, header, raw)
            label = "retagged+split" if split else "retagged"
        else:
            label = "already compatible"
        print(f"[{i}/{len(shards)}] {shard}: {label}", flush=True)

    # weight map: individual expert keys replace the stacked ones (each lives
    # in the shard its stacked tensor was split from)
    stacked_shard = {}
    for key in list(weight_map):
        m = re.match(r"((?:layers\.\d+|mtp\.\d+)\.ffn\.experts)\."
                     r"(gate_proj|up_proj|down_proj)\.(weight|scales)$", key)
        if m and key in dtypes:
            stacked_shard[key] = weight_map[key]
            del weight_map[key]
    for key, n in all_splits.items():
        m = re.match(r"((?:layers\.\d+|mtp\.\d+)\.ffn\.experts)\."
                     r"(gate_proj|up_proj|down_proj)\.(weight|scales)$", key)
        if not m or key[:-(len(m.group(3)) + 1)] not in modules:
            continue
        wn = _WN_OF[m.group(2)]
        base, suf = m.group(1), m.group(3)
        for e in range(n):
            weight_map[f"{base}.{e}.{wn}.{suf}"] = stacked_shard[key]

    with open(cfg_path + ".header.bak", "w") as f:
        json.dump(cfg, f, indent=2)
    for section_name in ("quantization", "quantization_config"):
        section = cfg.get(section_name)
        if isinstance(section, dict):
            section.pop("mode", None)   # omlx merges root mode into affine specs
            for path, entry in (section.get("modules") or {}).items():
                section.setdefault(path, dict(entry))   # omlx reads flat overrides
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print("config.json: root quantization mode dropped (module map kept); "
          "weight map: individual expert keys")
    return {"shards": len(shards), "modules": len(dtypes) // 2}


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Retag a stored mxfp4 MLX build in place for omlx's "
                    "deepseek_v41 source reader (bytes untouched).")
    p.add_argument("--build", required=True, help="build directory (modified in place)")
    p.add_argument("--dry-run", action="store_true", help="report and exit")
    a = p.parse_args()
    shim_build(a.build, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
