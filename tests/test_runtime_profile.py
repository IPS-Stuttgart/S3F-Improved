import csv
import json

import numpy as np

from se3plusplus_s3f.s1r2.relaxed_s3f_pilot import PilotConfig
from se3plusplus_s3f.s1r2.runtime_profile import (
    RuntimeProfileConfig,
    run_s3f_runtime_profile,
    write_s3f_runtime_profile_outputs,
)


def test_runtime_profile_smoke_outputs_phase_metrics(tmp_path):
    config = RuntimeProfileConfig(
        pilot=PilotConfig(
            grid_sizes=(8,),
            variants=("r1_r2",),
            n_trials=1,
            n_steps=2,
        )
    )

    rows = run_s3f_runtime_profile(config)

    assert len(rows) == 1
    row = rows[0]
    assert row["grid_size"] == 8
    assert row["variant"] == "r1_r2"
    assert float(row["total_wall_ms_per_step"]) > 0.0
    assert float(row["predict_linear_ms_per_step"]) > 0.0
    assert float(row["update_ms_per_step"]) > 0.0
    assert np.isfinite(float(row["position_rmse"]))

    outputs = write_s3f_runtime_profile_outputs(tmp_path / "out", config, write_plots=False)
    required_outputs = {name: outputs[name] for name in {"metrics", "metadata", "note"}}
    assert set(required_outputs) == {"metrics", "metadata", "note"}
    assert all(path.is_file() for path in required_outputs.values())

    with required_outputs["metrics"].open(newline="", encoding="utf-8") as metrics_file:
        assert sum(1 for _ in csv.DictReader(metrics_file)) == 1

    metadata_text = required_outputs["metadata"].read_text(encoding="utf-8")
    assert {"experiment": "s3f_runtime_profile", "metrics_rows": 1}.items() <= json.loads(metadata_text).items()
