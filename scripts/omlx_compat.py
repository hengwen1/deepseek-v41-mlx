"""Retag a stored mxfp4 MLX build in place for omlx's deepseek_v41 source reader.

    python scripts/omlx_compat.py --build /path/to/build [--dry-run]

Byte-packed mxfp4 (weights U32-viewed-as-U8, scales retagged F8_E8M0) is what
omlx's native fp4 branch expects; the tensor bytes are untouched, only shard
headers are rewritten (originals kept as <shard>.header.bak). Builds with
mxfp8 modules are refused — re-convert those with convert_cli.py --omlx-compat.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from deepseek_v41_mlx.omlx_compat import shim_build


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", required=True, help="build directory (modified in place)")
    p.add_argument("--dry-run", action="store_true", help="report and exit")
    a = p.parse_args()
    shim_build(a.build, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
