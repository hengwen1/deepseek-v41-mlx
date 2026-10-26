"""Convert the raw DeepSeek-V4.1-Flash release to an MLX build.

    python scripts/convert_cli.py --src /path/DeepSeek-V4.1-Flash-src --dst out-mixed \
        --bits 8 --expert-bits 4 --engram-bits 4

Recommended for 512 GB machines: --bits 8 --expert-bits 4 --engram-bits 4
(~427 GB). Without --engram-bits the two native-fp8 engram tables alone are
203 GB and no expert width fits under 512 GB.

Every bits option takes an affine bit width (2-8) or a microscaling format,
`mxfp4`/`mxfp8` (`mxf4`/`mxf8` accepted): E2M1/E4M3 elements sharing one e8m0
scale per group of 32 and no bias term, so the group size is pinned to 32 for
those modules.

    python scripts/convert_cli.py --src <src> --dst <out> --bits mxfp4
    python scripts/convert_cli.py --src <src> --dst <out> --bits 8 --expert-bits mxfp4  # mixed

The MTP draft stack (`mtp.*`, 3 stages) is kept by default — sanitized and
quantized like the rest (its routed experts are fp4 too) and carried by the
build as declared passthrough; pass --no-mtp to drop it (earlier builds did).

    python scripts/convert_cli.py --src <src> --dst <out> --bits 8 --no-mtp

--omlx-compat targets omlx's deepseek_v41 source reader: mxfp4 weights are
stored byte-packed (U8 + F8_E8M0 scales — the same words, which omlx's native
fp4 branch consumes losslessly) and mxfp8 modules become affine 8-bit (no
mxfp8 source path there). Existing mxfp4-only builds can be converted without
re-quantizing via scripts/omlx_compat.py.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from deepseek_v41_mlx.convert import convert, fmt_bits, parse_bits, quant_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--bits", type=parse_bits, default=None, metavar="BITS",
                   help="affine bit width (2-8) or mxfp4/mxfp8; default for every family")
    p.add_argument("--expert-bits", type=parse_bits, default=None, metavar="BITS",
                   help="routed experts; defaults to --bits")
    p.add_argument("--engram-bits", type=parse_bits, default=None, metavar="BITS",
                   help="engram tables; defaults to native fp8+ue8m0 passthrough")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--no-mtp", action="store_true",
                   help="drop the MTP draft stack tensors (mtp.*)")
    p.add_argument("--omlx-compat", action="store_true",
                   help="store mxfp4 byte-packed (U8 + F8_E8M0) for omlx's "
                        "deepseek_v41 reader; mxfp8 modules become affine 8-bit")
    p.add_argument("--resume", action="store_true",
                   help="skip groups whose output shard + manifest already exist")
    a = p.parse_args()
    if a.bits is not None or a.engram_bits is not None:
        expert = a.expert_bits if a.expert_bits is not None else a.bits
        parts = [f"experts {fmt_bits(quant_params(expert, a.group_size))}"]
        if a.bits is not None:
            parts.append(f"other {fmt_bits(quant_params(a.bits, a.group_size))}")
        if a.engram_bits is not None:
            parts.append(f"engram {fmt_bits(quant_params(a.engram_bits, a.group_size))}")
        print(f"recipe: {', '.join(parts)}; mtp {'dropped' if a.no_mtp else 'kept'}"
              + ("; omlx-compat" if a.omlx_compat else ""), flush=True)
    t0 = time.time()
    acct = convert(a.src, a.dst, bits=a.bits, expert_bits=a.expert_bits,
                   engram_bits=a.engram_bits, group_size=a.group_size,
                   keep_mtp=not a.no_mtp, omlx_compat=a.omlx_compat,
                   resume=a.resume)
    print(f"done in {time.time() - t0:.0f}s: {acct}")


if __name__ == "__main__":
    main()
