"""Command-line entry points for SE3PlusPlusS3F experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .wp1.relaxed_s3f_pilot import PilotConfig, write_relaxed_s3f_pilot_outputs


def main() -> None:
    args = _parse_args()
    if args.command == "wp1-relaxed-s3f":
        config = PilotConfig(
            grid_sizes=tuple(args.grid_sizes),
            n_trials=args.trials,
            n_steps=args.steps,
            seed=args.seed,
        )
        outputs = write_relaxed_s3f_pilot_outputs(
            output_dir=args.output_dir,
            config=config,
            write_plots=not args.no_plots,
        )
        for label, path in outputs.items():
            print(f"Wrote {label}: {path}")
        return

    raise ValueError(f"Unknown command {args.command!r}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    wp1 = subparsers.add_parser(
        "wp1-relaxed-s3f",
        help="Run the WP1 S1 x R2 relaxed S3F pilot benchmark.",
    )
    wp1.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "wp1_s1_r2_relaxed_s3f",
    )
    wp1.add_argument("--grid-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    wp1.add_argument("--trials", type=int, default=48)
    wp1.add_argument("--steps", type=int, default=24)
    wp1.add_argument("--seed", type=int, default=7)
    wp1.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
