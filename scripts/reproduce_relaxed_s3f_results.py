"""Reproduce the relaxed-S3F pilot outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from se3plusplus_s3f.s1r2.relaxed_s3f_pilot import load_pilot_config, write_relaxed_s3f_pilot_outputs  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "relaxed_s3f_pilot.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "relaxed_s3f_pilot"


def main() -> None:
    args = _parse_args()
    config = load_pilot_config(args.config)
    outputs = write_relaxed_s3f_pilot_outputs(
        output_dir=args.output_dir,
        config=config,
        write_plots=not args.no_plots,
    )
    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
