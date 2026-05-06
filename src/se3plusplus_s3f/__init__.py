"""Experiment code for multiresolution S3F on SE(3)++."""

from .s1r2.baseline_comparison import BaselineComparisonConfig, ParticleSensitivityConfig, run_baseline_comparison, run_particle_sensitivity
from .s1r2.relaxed_s3f_pilot import PilotConfig, load_pilot_config, run_relaxed_s3f_pilot
from .s1r2.euroc_planar import EuRoCPlanarConfig, run_euroc_planar_relaxed_s3f
from .s1r2.highres_reference import HighResReferenceConfig, run_highres_reference_benchmark
from .s1r2.quality_cost import QualityCostConfig, QualityCostResult, run_quality_cost_report
from .s1r2.runtime_profile import RuntimeProfileConfig, run_s3f_runtime_profile
from .s3r3.evidence_summary import S3R3EvidenceSummaryConfig, S3R3EvidenceSummaryResult, run_s3r3_evidence_summary
from .s3r3.dynamic_pose import S3R3DynamicPoseConfig, S3R3DynamicPoseResult, run_s3r3_dynamic_pose_benchmark
from .s3r3.dynamic_robustness import S3R3DynamicRobustnessConfig, S3R3DynamicRobustnessResult, run_s3r3_dynamic_robustness_sweep
from .s3r3.dynamic_highres_reference import S3R3DynamicHighResReferenceConfig, S3R3DynamicHighResReferenceResult, run_s3r3_dynamic_highres_reference_benchmark
from .s3r3.euroc_comparison_report import EuRoCS3R3ComparisonReportConfig, EuRoCS3R3ComparisonReportResult, run_euroc_s3r3_comparison_report
from .s3r3.euroc_pose import EuRoCS3R3PoseConfig, EuRoCS3R3PoseResult, run_euroc_s3r3_pose
from .s3r3.highres_reference import S3R3HighResReferenceConfig, run_s3r3_highres_reference_benchmark
from .s3r3.manifold_ukf import SO3R3ManifoldUKFConfig, make_so3r3_manifold_ukf
from .s3r3.orientation_basis import S3R3OrientationBasisConfig, run_s3r3_orientation_basis_diagnostic
from .s3r3.particle_comparison import S3R3ParticleComparisonConfig, S3R3ParticleComparisonResult, run_s3r3_particle_comparison
from .s3r3.relaxed_s3f_prototype import S3R3PrototypeConfig, run_s3r3_relaxed_prototype
from .s3r3.stress_sweep import S3R3StressSweepConfig, S3R3StressSweepResult, run_s3r3_stress_sweep

__all__ = [
    "BaselineComparisonConfig",
    "EuRoCPlanarConfig",
    "EuRoCS3R3ComparisonReportConfig",
    "EuRoCS3R3ComparisonReportResult",
    "EuRoCS3R3PoseConfig",
    "EuRoCS3R3PoseResult",
    "HighResReferenceConfig",
    "ParticleSensitivityConfig",
    "PilotConfig",
    "QualityCostConfig",
    "QualityCostResult",
    "RuntimeProfileConfig",
    "S3R3DynamicPoseConfig",
    "S3R3DynamicPoseResult",
    "S3R3DynamicRobustnessConfig",
    "S3R3DynamicRobustnessResult",
    "S3R3DynamicHighResReferenceConfig",
    "S3R3DynamicHighResReferenceResult",
    "S3R3EvidenceSummaryConfig",
    "S3R3EvidenceSummaryResult",
    "S3R3HighResReferenceConfig",
    "SO3R3ManifoldUKFConfig",
    "S3R3OrientationBasisConfig",
    "S3R3ParticleComparisonConfig",
    "S3R3ParticleComparisonResult",
    "S3R3PrototypeConfig",
    "S3R3StressSweepConfig",
    "S3R3StressSweepResult",
    "load_pilot_config",
    "make_so3r3_manifold_ukf",
    "run_baseline_comparison",
    "run_euroc_planar_relaxed_s3f",
    "run_euroc_s3r3_comparison_report",
    "run_euroc_s3r3_pose",
    "run_highres_reference_benchmark",
    "run_particle_sensitivity",
    "run_quality_cost_report",
    "run_relaxed_s3f_pilot",
    "run_s3r3_evidence_summary",
    "run_s3r3_dynamic_pose_benchmark",
    "run_s3r3_dynamic_robustness_sweep",
    "run_s3r3_dynamic_highres_reference_benchmark",
    "run_s3r3_highres_reference_benchmark",
    "run_s3r3_orientation_basis_diagnostic",
    "run_s3r3_particle_comparison",
    "run_s3r3_relaxed_prototype",
    "run_s3r3_stress_sweep",
    "run_s3f_runtime_profile",
]
