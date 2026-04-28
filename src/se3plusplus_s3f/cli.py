"""Command-line entry points for SE3PlusPlusS3F experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .wp1.euroc_planar import EuRoCPlanarConfig, write_euroc_planar_outputs
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

    if args.command == "wp1-euroc-planar":
        config = EuRoCPlanarConfig(
            grid_size=args.grid_size,
            start_index=args.start_index,
            stride=args.stride,
            max_steps=args.steps,
            seed=args.seed,
            measurement_noise_std=args.measurement_noise_std,
            process_noise_std=args.process_noise_std,
            initial_position_std=args.initial_position_std,
            orientation_prior_kappa=args.orientation_prior_kappa,
        )
        outputs = write_euroc_planar_outputs(
            groundtruth_path=args.groundtruth_path,
            output_dir=args.output_dir,
            config=config,
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

    euroc = subparsers.add_parser(
        "wp1-euroc-planar",
        help="Run a planar relaxed S3F smoke test on EuRoC ground truth.",
    )
    euroc.add_argument("--groundtruth-path", type=Path, required=True)
    euroc.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "wp1_euroc_planar",
    )
    euroc.add_argument("--grid-size", type=int, default=16)
    euroc.add_argument("--start-index", type=int, default=0)
    euroc.add_argument("--stride", type=int, default=20)
    euroc.add_argument("--steps", type=int, default=60)
    euroc.add_argument("--seed", type=int, default=13)
    euroc.add_argument("--measurement-noise-std", type=float, default=0.05)
    euroc.add_argument("--process-noise-std", type=float, default=0.01)
    euroc.add_argument("--initial-position-std", type=float, default=0.08)
    euroc.add_argument("--orientation-prior-kappa", type=float, default=6.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
