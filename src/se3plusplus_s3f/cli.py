"""Command-line entry points for SE3PlusPlusS3F experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .s1r2.baseline_comparison import (
    BaselineComparisonConfig,
    ParticleSensitivityConfig,
    write_baseline_comparison_outputs,
    write_particle_sensitivity_outputs,
)
from .s1r2.euroc_planar import EuRoCPlanarConfig, write_euroc_planar_outputs
from .s1r2.highres_reference import HighResReferenceConfig, write_highres_reference_outputs
from .s1r2.quality_cost import QUALITY_COST_VARIANTS, QualityCostConfig, write_quality_cost_outputs
from .s1r2.relaxed_s3f_pilot import PilotConfig, load_pilot_config, write_relaxed_s3f_pilot_outputs
from .s1r2.runtime_profile import RuntimeProfileConfig, write_s3f_runtime_profile_outputs
from .s3r3.highres_reference import S3R3HighResReferenceConfig, write_s3r3_highres_reference_outputs
from .s3r3.relaxed_s3f_prototype import S3R3PrototypeConfig, write_s3r3_relaxed_outputs


def main() -> None:
    args = _parse_args()
    if args.command == "relaxed-s3f":
        base_config = load_pilot_config(args.config) if args.config is not None else PilotConfig()
        config = replace(
            base_config,
            grid_sizes=tuple(args.grid_sizes) if args.grid_sizes is not None else base_config.grid_sizes,
            n_trials=args.trials if args.trials is not None else base_config.n_trials,
            n_steps=args.steps if args.steps is not None else base_config.n_steps,
            seed=args.seed if args.seed is not None else base_config.seed,
        )
        outputs = write_relaxed_s3f_pilot_outputs(
            output_dir=args.output_dir,
            config=config,
            write_plots=not args.no_plots,
        )
        for label, path in outputs.items():
            print(f"Wrote {label}: {path}")
        return

    if args.command == "highres-reference":
        config = HighResReferenceConfig(
            pilot=PilotConfig(
                grid_sizes=tuple(args.grid_sizes),
                n_trials=args.trials,
                n_steps=args.steps,
                seed=args.seed,
            ),
            reference_grid_size=args.reference_grid_size,
        )
        outputs = write_highres_reference_outputs(
            output_dir=args.output_dir,
            config=config,
            write_plots=not args.no_plots,
        )
        for label, path in outputs.items():
            print(f"Wrote {label}: {path}")
        return

    if args.command == "compare-baselines":
        config = BaselineComparisonConfig(
            pilot=PilotConfig(
                grid_sizes=tuple(args.grid_sizes),
                n_trials=args.trials,
                n_steps=args.steps,
                seed=args.seed,
            ),
            particle_count=args.particle_count,
            particle_seed=args.particle_seed,
        )
        outputs = write_baseline_comparison_outputs(
            output_dir=args.output_dir,
            config=config,
            write_plots=not args.no_plots,
        )
        for label, path in outputs.items():
            print(f"Wrote {label}: {path}")
        return

    if args.command == "quality-cost":
        config = QualityCostConfig(
            reference=HighResReferenceConfig(
                pilot=PilotConfig(
                    grid_sizes=tuple(args.grid_sizes),
                    variants=QUALITY_COST_VARIANTS,
                    n_trials=args.trials,
                    n_steps=args.steps,
                    seed=args.seed,
                ),
                reference_grid_size=args.reference_grid_size,
            ),
            particle_counts=tuple(args.particle_counts),
            particle_seed=args.particle_seed,
            repeats=args.repeats,
            repeat_seed_stride=args.repeat_seed_stride,
        )
        outputs = write_quality_cost_outputs(
            output_dir=args.output_dir,
            config=config,
            write_plots=not args.no_plots,
        )
        for label, path in outputs.items():
            print(f"Wrote {label}: {path}")
        return

    if args.command == "particle-sensitivity":
        config = ParticleSensitivityConfig(
            pilot=PilotConfig(
                grid_sizes=tuple(args.grid_sizes),
                n_trials=args.trials,
                n_steps=args.steps,
                seed=args.seed,
            ),
            particle_counts=tuple(args.particle_counts),
            particle_seed=args.particle_seed,
        )
        outputs = write_particle_sensitivity_outputs(
            output_dir=args.output_dir,
            config=config,
            write_plots=not args.no_plots,
        )
        for label, path in outputs.items():
            print(f"Wrote {label}: {path}")
        return

    if args.command == "profile-s3f-runtime":
        config = RuntimeProfileConfig(
            pilot=PilotConfig(
                grid_sizes=tuple(args.grid_sizes),
                variants=tuple(args.variants),
                n_trials=args.trials,
                n_steps=args.steps,
                seed=args.seed,
            )
        )
        outputs = write_s3f_runtime_profile_outputs(
            output_dir=args.output_dir,
            config=config,
            write_plots=not args.no_plots,
        )
        for label, path in outputs.items():
            print(f"Wrote {label}: {path}")
        return

    if args.command == "euroc-planar":
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

    if args.command == "s3r3-relaxed":
        config = S3R3PrototypeConfig(
            grid_sizes=tuple(args.grid_sizes),
            n_trials=args.trials,
            n_steps=args.steps,
            seed=args.seed,
            cell_sample_count=args.cell_sample_count,
        )
        outputs = write_s3r3_relaxed_outputs(
            output_dir=args.output_dir,
            config=config,
            write_plots=not args.no_plots,
        )
        for label, path in outputs.items():
            print(f"Wrote {label}: {path}")
        return

    if args.command == "s3r3-highres-reference":
        config = S3R3HighResReferenceConfig(
            prototype=S3R3PrototypeConfig(
                grid_sizes=tuple(args.grid_sizes),
                n_trials=args.trials,
                n_steps=args.steps,
                seed=args.seed,
                cell_sample_count=args.cell_sample_count,
            ),
            reference_grid_size=args.reference_grid_size,
        )
        outputs = write_s3r3_highres_reference_outputs(
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

    relaxed = subparsers.add_parser(
        "relaxed-s3f",
        help="Run the S1 x R2 relaxed S3F pilot benchmark.",
    )
    relaxed.add_argument("--config", type=Path, help="JSON config file for the pilot benchmark.")
    relaxed.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "relaxed_s3f_pilot",
    )
    relaxed.add_argument("--grid-sizes", type=int, nargs="+")
    relaxed.add_argument("--trials", type=int)
    relaxed.add_argument("--steps", type=int)
    relaxed.add_argument("--seed", type=int)
    relaxed.add_argument("--no-plots", action="store_true")

    highres = subparsers.add_parser(
        "highres-reference",
        help="Compare coarse relaxed S3F variants against a high-resolution S3F reference.",
    )
    highres.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "highres_reference",
    )
    highres.add_argument("--grid-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    highres.add_argument("--reference-grid-size", type=int, default=256)
    highres.add_argument("--trials", type=int, default=16)
    highres.add_argument("--steps", type=int, default=16)
    highres.add_argument("--seed", type=int, default=17)
    highres.add_argument("--no-plots", action="store_true")

    comparison = subparsers.add_parser(
        "compare-baselines",
        help="Compare S1 x R2 relaxed S3F variants against EKF and particle baselines.",
    )
    comparison.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "baseline_comparison",
    )
    comparison.add_argument("--grid-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    comparison.add_argument("--trials", type=int, default=32)
    comparison.add_argument("--steps", type=int, default=20)
    comparison.add_argument("--seed", type=int, default=7)
    comparison.add_argument("--particle-count", type=int, default=1024)
    comparison.add_argument("--particle-seed", type=int, default=101)
    comparison.add_argument("--no-plots", action="store_true")

    quality_cost = subparsers.add_parser(
        "quality-cost",
        help="Summarize relaxed S3F quality, consistency, runtime, and high-resolution reference error.",
    )
    quality_cost.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "quality_cost",
    )
    quality_cost.add_argument("--grid-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    quality_cost.add_argument("--reference-grid-size", type=int, default=256)
    quality_cost.add_argument("--trials", type=int, default=16)
    quality_cost.add_argument("--steps", type=int, default=16)
    quality_cost.add_argument("--seed", type=int, default=17)
    quality_cost.add_argument("--particle-counts", type=int, nargs="+", default=[128, 512, 2048, 8192])
    quality_cost.add_argument("--particle-seed", type=int, default=101)
    quality_cost.add_argument("--repeats", type=int, default=1)
    quality_cost.add_argument("--repeat-seed-stride", type=int, default=1000)
    quality_cost.add_argument("--no-plots", action="store_true")

    sensitivity = subparsers.add_parser(
        "particle-sensitivity",
        help="Sweep particle counts against S1 x R2 relaxed S3F grid sizes.",
    )
    sensitivity.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "particle_sensitivity",
    )
    sensitivity.add_argument("--grid-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    sensitivity.add_argument("--particle-counts", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 8192])
    sensitivity.add_argument("--trials", type=int, default=32)
    sensitivity.add_argument("--steps", type=int, default=20)
    sensitivity.add_argument("--seed", type=int, default=7)
    sensitivity.add_argument("--particle-seed", type=int, default=101)
    sensitivity.add_argument("--no-plots", action="store_true")

    profile = subparsers.add_parser(
        "profile-s3f-runtime",
        help="Profile relaxed S3F prediction and update runtime phases.",
    )
    profile.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "s3f_runtime_profile",
    )
    profile.add_argument("--grid-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    profile.add_argument("--variants", nargs="+", default=["baseline", "r1", "r1_r2"])
    profile.add_argument("--trials", type=int, default=16)
    profile.add_argument("--steps", type=int, default=16)
    profile.add_argument("--seed", type=int, default=7)
    profile.add_argument("--no-plots", action="store_true")

    euroc = subparsers.add_parser(
        "euroc-planar",
        help="Run a planar relaxed S3F smoke test on EuRoC ground truth.",
    )
    euroc.add_argument("--groundtruth-path", type=Path, required=True)
    euroc.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "euroc_planar",
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

    s3r3 = subparsers.add_parser(
        "s3r3-relaxed",
        help="Run the minimal S3+ x R3 relaxed S3F prototype benchmark.",
    )
    s3r3.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "s3r3_relaxed",
    )
    s3r3.add_argument("--grid-sizes", type=int, nargs="+", default=[16, 32])
    s3r3.add_argument("--trials", type=int, default=8)
    s3r3.add_argument("--steps", type=int, default=8)
    s3r3.add_argument("--seed", type=int, default=23)
    s3r3.add_argument("--cell-sample-count", type=int, default=27)
    s3r3.add_argument("--no-plots", action="store_true")

    s3r3_highres = subparsers.add_parser(
        "s3r3-highres-reference",
        help="Compare S3+ x R3 relaxed S3F variants against a denser S3F reference.",
    )
    s3r3_highres.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "s3r3_highres_reference",
    )
    s3r3_highres.add_argument("--grid-sizes", type=int, nargs="+", default=[8, 16, 32])
    s3r3_highres.add_argument("--reference-grid-size", type=int, default=64)
    s3r3_highres.add_argument("--trials", type=int, default=8)
    s3r3_highres.add_argument("--steps", type=int, default=8)
    s3r3_highres.add_argument("--seed", type=int, default=29)
    s3r3_highres.add_argument("--cell-sample-count", type=int, default=27)
    s3r3_highres.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
