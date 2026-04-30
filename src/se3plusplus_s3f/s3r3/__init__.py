"""S3+ x R3 relaxed S3F prototype experiments."""

from .highres_reference import (
    S3R3HighResReferenceConfig,
    run_s3r3_highres_reference_benchmark,
    write_s3r3_highres_reference_outputs,
)
from .relaxed_s3f_prototype import (
    S3R3PrototypeConfig,
    run_s3r3_relaxed_prototype,
    write_s3r3_relaxed_outputs,
)

__all__ = [
    "S3R3HighResReferenceConfig",
    "S3R3PrototypeConfig",
    "run_s3r3_highres_reference_benchmark",
    "run_s3r3_relaxed_prototype",
    "write_s3r3_highres_reference_outputs",
    "write_s3r3_relaxed_outputs",
]
