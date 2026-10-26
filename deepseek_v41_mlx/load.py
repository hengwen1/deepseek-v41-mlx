"""Load a converted DeepSeek-V4.1 MLX build — strictly.

``mlx_lm.load`` uses ``strict=False`` and lets missing weights silently stay
random-initialized (the GLM-5.2 lesson); this loader refuses that:

* every parameter the model owns must come from the checkpoint;
* every checkpoint tensor must either land in a parameter or belong to the
  explicitly-declared vision passthrough (``vision.*`` / ``aligner.*`` /
  ``image_*`` — carried by the conversion for a future VL runtime, reported,
  never silently ignored);
* quantization is replayed from ``config.json``: the converter writes an
  explicit per-module map (``quantization`` carries defaults plus
  ``<module path>: {group_size, bits[, mode]} | false`` entries keyed by
  runtime-internal module paths; ``mode`` selects mxfp4/mxfp8 microscaling —
  no biases, group size 32); when the map is absent the loader falls back
  to the same ``is_quant_target`` / ``bits_for`` rules the converter used.
* the MTP draft stack (``mtp.*``), when the build retained it, is declared
  passthrough like the vision tower: reported, never loaded — the text-only
  runtime has no DSpark module.

Builds bundle **nothing executable**: no ``model_file`` mechanism exists here —
a build is loaded through this package (``deepseek_v41_mlx.load.load``), never
through ``mlx_lm.load``.
"""

from __future__ import annotations

import glob
import json
import os

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import ModelArgs
from .convert import VISION_PREFIXES, bits_for, is_quant_target
from .model import Model


def quant_predicate(group_size: int, bits, expert_bits=None,
                    module_map: dict | None = None):
    """Per-module predicate. With ``module_map`` (the converter's explicit map,
    keyed by runtime-internal module paths) the map is authoritative — entries
    may carry a microscaling ``mode``; without it, fall back to the converter's
    rule set. ``bits``/``expert_bits`` take affine widths or complete param
    dicts, matching both old int-only and current mode-aware config blocks."""
    expert = expert_bits if isinstance(expert_bits, dict) else \
        ({"group_size": group_size, "bits": int(expert_bits)} if expert_bits else None)

    def pred(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if module_map is not None:
            entry = module_map.get(path, False)
            return dict(entry) if entry else False
        w = getattr(module, "weight", None)
        if w is None:
            return False
        name = path + ".weight"
        if not is_quant_target(name, w.shape[-1], group_size):
            return False
        return bits_for(name, {"group_size": group_size, "bits": bits},
                        expert, group_size)
    return pred


def load_token_map(path: str, args: ModelArgs, cache_dir: str | None = None):
    """The engram compressed-token map, built from the release tokenizer.

    Cached (building it decodes+normalizes all 129,280 tokens, ~a minute) in
    ``cache_dir`` when given — use that when ``path`` must stay read-only —
    else beside the tokenizer. Returns None when no tokenizer is present.
    """
    cache_file = os.path.join(cache_dir or path, "engram_token_map.json")
    if os.path.exists(cache_file):
        return json.load(open(cache_file))
    tok_file = os.path.join(path, "tokenizer.json")
    if not os.path.exists(tok_file):
        return None
    from transformers import PreTrainedTokenizerFast
    from .engram import build_compressed_token_map
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    token_map, n = build_compressed_token_map(tok)
    if args.engram_compressed_vocab_size and n != args.engram_compressed_vocab_size:
        raise ValueError(f"compressed vocab {n} != config {args.engram_compressed_vocab_size}")
    json.dump(token_map, open(cache_file, "w"))
    return token_map


def load(path: str, lazy: bool | None = None, strict: bool = True):
    """``lazy=None`` auto-selects: builds that fit comfortably in physical RAM
    are materialized at load time (CPU-side, avoiding cold-mmap page-in inside
    a GPU command buffer — that trips Metal's watchdog); builds larger than
    ~80% of RAM stay mmap-lazy, because materialized buffers are unevictable
    and drive the machine into compressor thrash, while clean file-backed
    pages stream from disk per forward (measured: a 476 GB build on 512 GiB
    stalled the box when materialized, runs lazily)."""
    try:
        mx.set_wired_limit(int(float(os.environ.get("DSV41_WIRED_GB", "470")) * 1e9))
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("DSV41_LAZY") in ("0", "1"):
        lazy = os.environ["DSV41_LAZY"] == "1"
        print(f"[load] lazy={lazy} forced via DSV41_LAZY", flush=True)
    if lazy is None:
        size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(path, "*.safetensors")))
        try:
            phys = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            phys = int(550e9)
        lazy = size > 0.8 * phys
        if lazy:
            print(f"[load] {size / 1e9:.0f} GB build vs {phys / 1e9:.0f} GB RAM: "
                  f"staying mmap-lazy (materializing would thrash the compressor)",
                  flush=True)
            # Prewarm the page cache sequentially (clean, evictable pages) so
            # the first forward's page-ins are cache hits — cold-disk faults
            # inside a GPU command buffer trip Metal's ~watchdog.
            import time as _time
            t0 = _time.time()
            buf = bytearray(1 << 28)
            for p in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
                with open(p, "rb", buffering=0) as fh:
                    while fh.readinto(buf):
                        pass
            print(f"[load] page-cache prewarm: {size / 1e9:.0f} GB in "
                  f"{_time.time() - t0:.0f}s", flush=True)
    cfg = json.load(open(os.path.join(path, "config.json")))
    args = ModelArgs.from_dict(cfg)
    token_map = load_token_map(path, args) if args.engram_layer_ids else None
    model = Model(args, token_map=token_map)

    q = cfg.get("quantization")
    if q:
        if q.get("bits"):
            module_map = q.get("modules")
            nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                        class_predicate=quant_predicate(q["group_size"], q["bits"],
                                                        q.get("expert_bits"),
                                                        module_map))
        if q.get("engram_bits"):
            from .engram import QuantizedEngramEmbedding
            eb = q["engram_bits"]
            if not isinstance(eb, dict):
                eb = {"group_size": q["group_size"], "bits": int(eb)}
            for layer in model.layers:
                if layer.engram is not None:
                    e = layer.engram.embed
                    layer.engram.embed = QuantizedEngramEmbedding(
                        e.weight.shape[0], e.weight.shape[1],
                        eb["group_size"], eb["bits"], mode=eb.get("mode", "affine"))

    loaded, vision_skipped, mtp_skipped = set(), set(), set()
    for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        w = mx.load(shard)
        items = []
        for k, v in w.items():
            if k.startswith(VISION_PREFIXES):
                vision_skipped.add(k)      # declared passthrough, text-only runtime
                continue
            if k.startswith("mtp."):
                mtp_skipped.add(k)         # declared passthrough, draft stack
                continue
            items.append((k, v))
        model.load_weights(items, strict=False)
        if not lazy:
            mx.eval([v for _, v in items])
        loaded.update(k for k, _ in items)
        del w

    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing = expected - loaded
    unexpected = loaded - expected
    if strict and missing:
        raise ValueError(f"{len(missing)} params missing from checkpoint, "
                         f"e.g. {sorted(missing)[:4]}")
    if strict and unexpected:
        raise ValueError(f"{len(unexpected)} checkpoint tensors have no home, "
                         f"e.g. {sorted(unexpected)[:4]}")
    model.eval()
    model._vision_passthrough = sorted(vision_skipped)
    model._mtp_passthrough = sorted(mtp_skipped)
    return model, args
