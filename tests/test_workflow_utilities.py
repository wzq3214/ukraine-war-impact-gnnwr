from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "reproducibility"
    / "section_3_4_gnnwr_official_workflow.py"
)
spec = importlib.util.spec_from_file_location("section34_workflow", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_logit_round_trip() -> None:
    values = np.array([1e-5, 0.01, 0.2, 0.8, 0.999])
    transformed = module.logit_transform(values, 1e-6)
    assert np.allclose(expit(transformed), values)


def test_block_assignment_preserves_complete_blocks() -> None:
    blocks = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4])
    mapping = module.assign_blocks_to_folds(blocks, n_folds=3, seed=42)
    assert set(mapping) == set(np.unique(blocks))
    assert all(0 <= fold < 3 for fold in mapping.values())


def test_metrics_are_exact_for_perfect_prediction() -> None:
    observed = np.array([-2.0, 0.0, 3.0])
    metrics = module.calculate_metrics(observed, observed.copy())
    assert metrics["R2"] == 1.0
    assert metrics["RMSE"] == 0.0
    assert metrics["MAE"] == 0.0
    assert metrics["Bias"] == 0.0
