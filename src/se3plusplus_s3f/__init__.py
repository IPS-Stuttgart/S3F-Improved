"""Experiment code for multiresolution S3F on SE(3)++."""

from .s1r2.baseline_comparison import BaselineComparisonConfig, ParticleSensitivityConfig, run_baseline_comparison, run_particle_sensitivity
from .s1r2.relaxed_s3f_pilot import PilotConfig, load_pilot_config, run_relaxed_s3f_pilot
from .s1r2.euroc_planar import EuRoCPlanarConfig, run_euroc_planar_relaxed_s3f
from .s1r2.highres_reference import HighResReferenceConfig, run_highres_reference_benchmark

__all__ = [
    "BaselineComparisonConfig",
    "EuRoCPlanarConfig",
    "HighResReferenceConfig",
    "ParticleSensitivityConfig",
    "PilotConfig",
    "load_pilot_config",
    "run_baseline_comparison",
    "run_euroc_planar_relaxed_s3f",
    "run_highres_reference_benchmark",
    "run_particle_sensitivity",
    "run_relaxed_s3f_pilot",
]
