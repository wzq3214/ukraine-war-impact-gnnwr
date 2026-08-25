#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Reproduce the GNNWR analysis reported in Section 3.4 of the Ukraine study.

The script is an analysis driver for the official open-source GNNWR package.
It does not reimplement, simplify or replace the upstream GNNWR algorithm and
it does not use a Tiny-GNNWR surrogate. The upstream source code should remain
unchanged. This workflow pins the public GNNWR package to version 0.1.17
and records the installed version and source location in the run manifest.

Default analytical fields in the supplied workbook
--------------------------------------------------
Response       Disp_Rate_FINNAL
CII predictor  CII_pos
Older adults   Index_Agin
Damage         ALL_Damage_Density
NTL extinction Extinguished_Ratio
Coordinates    proj_X and proj_Y
ADM3 key       ADM3_PCODE

Main outputs
------------
1. Five-fold spatial-block out-of-fold predictions for GNNWR, GWR and OLS.
2. Table 3 with R2, RMSE, MAE and residual Global Moran's I.
3. A final full-sample GNNWR fit and standardized local coefficients.
4. Fig. 6 using UKR_ADM3_2023.shp. Coefficients are mapped only where they
   were estimated. No zero filling or spatial interpolation is performed.
5. Data, fold, tuning, coefficient and software-environment audit files.

Scientific safeguards
---------------------
* The analytical sample is defined by valid positive response records and
  complete projected coordinates. No fixed sample count is hard-coded.
* CII_pos is used as the computational CII field. When CII is also present,
  the script verifies whether CII_pos differs from CII only by an additive
  constant and records the result.
* Imputation, standardisation and model selection are fitted only on the
  corresponding training data.
* GNNWR, GWR and OLS use identical outer spatial folds.
* Table 3 is calculated from pooled spatially out-of-fold predictions on the
  logit-transformed response scale.
* Local coefficients are conditional spatial associations, not significance
  tests, causal effects or local mechanisms.

Upstream references
-------------------
Du, Z., Wang, Z., Wu, S., Zhang, F., and Liu, R. 2020. Geographically neural
network weighted regression for the accurate estimation of spatial
non-stationarity. International Journal of Geographical Information Science,
34, 1353-1377. https://doi.org/10.1080/13658816.2019.1707834

Yin, Z. et al. 2024. GNNWR: an open-source package of spatiotemporal intelligent
regression methods for modeling spatial and temporal nonstationarity.
Geoscientific Model Development, 17, 8455-8468.
https://doi.org/10.5194/gmd-17-8455-2024

Pinned upstream dependency
--------------------------
GNNWR 0.1.17. The corresponding documented upstream revision is
2711bdee9821bfe242992d2eb885a78be1cf2628.

SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import inspect
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "reproducibility" else SCRIPT_DIR
UPSTREAM_GNNWR_COMMIT = "2711bdee9821bfe242992d2eb885a78be1cf2628"
GNNWR_PAPER_DOI = "10.1080/13658816.2019.1707834"
GNNWR_PACKAGE_DOI = "10.5194/gmd-17-8455-2024"

_GNNWR_DATASETS: Any = None
_GNNWR_MODELS: Any = None


@dataclass
class Config:
    """Complete run configuration recorded in the output manifest."""

    data_path: str
    output_dir: str

    # Data fields in the supplied Excel workbook
    external_id_col: str = "ADM3_PCODE"
    model_id_col: str = "__model_id"
    ratio_col: str = "Disp_Rate_FINNAL"
    target_col: str = "IDP_stock_ratio_logit"
    ratio_scale: str = "proportion"
    spatial_cols: Tuple[str, str] = ("proj_X", "proj_Y")
    display_cols: Tuple[str, str] = ("POINT_X", "POINT_Y")
    predictor_cols: Tuple[str, ...] = (
        "CII_pos",
        "Index_Agin",
        "ALL_Damage_Density",
        "Extinguished_Ratio",
    )
    coefficient_labels: Tuple[str, ...] = (
        "CII",
        "Older adult share",
        "Reported infrastructure damage density",
        "Nighttime light extinction ratio",
    )

    # Optional explicit record-selection and Fig. 6a inputs
    match_col: Optional[str] = None
    matched_ids_path: Optional[str] = None
    matched_id_col: Optional[str] = None
    input_is_matched_sample: bool = False
    stock_col: Optional[str] = None
    stock_table_path: Optional[str] = None
    stock_table_id_col: Optional[str] = None
    population_col: Optional[str] = None
    expected_stock_total: Optional[float] = None
    ratio_identity_tolerance: float = 1e-4
    logit_epsilon: float = 1e-6

    # Spatial validation
    n_spatial_blocks: int = 25
    n_spatial_folds: int = 5
    inner_validation_fraction: float = 0.20
    min_outer_fold_size: int = 15
    min_inner_validation_size: int = 12
    random_seed: int = 42

    # GNNWR specification used in the manuscript
    gnnwr_dense_layers: Tuple[int, ...] = (64, 32)
    gnnwr_start_lr: float = 0.02
    gnnwr_optimizer: str = "Adadelta"
    gnnwr_activation_init: float = 0.10
    gnnwr_drop_out: float = 0.20
    gnnwr_batch_norm: bool = True
    gnnwr_max_epoch: int = 3000
    gnnwr_early_stop: int = 300
    gnnwr_weight_decay: float = 1e-3
    final_seeds: Tuple[int, ...] = (42,)
    map_coefficient_estimator: str = "primary"

    # GWR benchmark
    gwr_min_neighbors: int = 20
    gwr_candidate_count: int = 10
    gwr_ridge: float = 1e-8

    # Residual Global Moran's I
    moran_k: int = 4
    moran_permutations: int = 999

    # Fig. 6
    boundary_path: Optional[str] = None
    boundary_layer: Optional[str] = None
    boundary_id_col: Optional[str] = None
    oblast_col: Optional[str] = None
    occupied_path: Optional[str] = None
    occupied_layer: Optional[str] = None
    map_crs: str = "EPSG:3035"
    figure_dpi: int = 600
    skip_figure: bool = False

    # Reproducibility
    expected_gnnwr_version: str = "0.1.17"
    recommended_gnnwr_commit: str = UPSTREAM_GNNWR_COMMIT
    require_gnnwr_version: bool = True


@dataclass
class FoldPreprocessor:
    """Training-only imputation and standardisation parameters."""

    imputer: SimpleImputer
    x_scaler: StandardScaler
    y_mean: float
    y_std: float

    def impute_x(self, frame: pd.DataFrame, predictors: Sequence[str]) -> np.ndarray:
        return self.imputer.transform(frame.loc[:, list(predictors)]).astype(np.float64)

    def transform_x(self, frame: pd.DataFrame, predictors: Sequence[str]) -> np.ndarray:
        return self.x_scaler.transform(self.impute_x(frame, predictors)).astype(np.float64)

    def transform_y(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        return (values - self.y_mean) / self.y_std

    def inverse_y(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        return values * self.y_std + self.y_mean


# =============================================================================
# General utilities
# =============================================================================


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def import_gnnwr() -> Tuple[Any, Any]:
    """Import the local upstream package only when model fitting is requested."""

    global _GNNWR_DATASETS, _GNNWR_MODELS
    if _GNNWR_DATASETS is not None and _GNNWR_MODELS is not None:
        return _GNNWR_DATASETS, _GNNWR_MODELS

    candidate_paths = (
        REPO_ROOT,
        REPO_ROOT / "src",
        REPO_ROOT / "vendor" / "gnnwr" / "src",
        SCRIPT_DIR,
    )
    for candidate in candidate_paths:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    attempts: List[str] = []
    for datasets_name, models_name in (
        ("src.gnnwr.datasets", "src.gnnwr.models"),
        ("gnnwr.datasets", "gnnwr.models"),
    ):
        try:
            _GNNWR_DATASETS = importlib.import_module(datasets_name)
            _GNNWR_MODELS = importlib.import_module(models_name)
            return _GNNWR_DATASETS, _GNNWR_MODELS
        except Exception as exc:
            attempts.append(f"{datasets_name}: {exc}")

    raise ImportError(
        "Cannot import GNNWR. Install the pinned package with `pip install "
        "gnnwr==0.1.17`, or provide an unmodified checkout under "
        "vendor/gnnwr. Import attempts were:\n" + "\n".join(attempts)
    )


def safe_read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Excel input without silently changing column names."""

    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, engine="openpyxl")
    if suffix == ".xls":
        return pd.read_excel(path)
    if suffix in {".csv", ".txt"}:
        errors: List[str] = []
        for encoding in ("utf-8-sig", "utf-8", "gbk"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except Exception as exc:
                errors.append(f"{encoding}: {exc}")
        raise RuntimeError("Unable to read delimited input.\n" + "\n".join(errors))
    raise ValueError(f"Unsupported input format: {suffix}")


def require_columns(df: pd.DataFrame, columns: Iterable[Optional[str]], label: str) -> None:
    required = [column for column in columns if column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def make_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out[list(columns)] = out[list(columns)].replace([np.inf, -np.inf], np.nan)
    return out


def normalise_key(series: pd.Series) -> pd.Series:
    """Create stable string keys for tabular-to-boundary joins."""

    text = series.astype("string").str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    return text.str.casefold()


def truthy_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.eq(1)
    text = series.astype("string").str.strip().str.casefold()
    text_mask = text.isin({"true", "yes", "y", "matched", "origin", "1"})
    return (numeric_mask | text_mask).fillna(False)


def logit_transform(values: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(values)):
        raise ValueError("The IDP stock ratio contains non-finite values.")
    if np.any(values <= 0) or np.any(values >= 1):
        invalid = values[(values <= 0) | (values >= 1)]
        raise ValueError(
            "The retained IDP stock ratios must be strictly between 0 and 1. "
            f"Invalid examples: {invalid[:10]}"
        )
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred have different shapes.")
    if np.any(~np.isfinite(y_true)) or np.any(~np.isfinite(y_pred)):
        raise ValueError("Metrics cannot be calculated from NaN or infinite values.")
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def tensor_or_number_to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().numpy().reshape(-1)[0])
    try:
        return float(value)
    except Exception:
        return float("nan")


def close_gnnwr_writer(model: Any) -> None:
    writer = getattr(model, "_writer", None)
    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_input_dataset(path: Optional[Path]) -> Mapping[str, str]:
    if path is None or not path.exists():
        return {}
    if path.suffix.lower() == ".shp":
        hashes: Dict[str, str] = {}
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qmd"):
            sidecar = path.with_suffix(suffix)
            if sidecar.exists():
                hashes[sidecar.name] = sha256_file(sidecar)
        return hashes
    return {path.name: sha256_file(path)}


def run_git_command(directory: Path, *args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return completed.stdout.strip()
    except Exception:
        return None


# =============================================================================
# Data selection and audit
# =============================================================================


def load_matched_id_keys(cfg: Config) -> Optional[set[str]]:
    if not cfg.matched_ids_path:
        return None
    path = Path(cfg.matched_ids_path).expanduser().resolve()
    matched = safe_read_table(path)
    column = cfg.matched_id_col or cfg.external_id_col
    require_columns(matched, [column], path.name)
    keys = set(normalise_key(matched[column]).dropna().tolist())
    if not keys:
        raise ValueError(f"No matched IDs were read from {path}.")
    if len(keys) != len(matched):
        warnings.warn(
            "The matched-ID file contains missing or duplicated identifiers. "
            "Only unique non-missing keys will be used.",
            RuntimeWarning,
        )
    return keys


def prepare_model_data(raw: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the workbook and construct the rule-defined modelling sample."""

    required = [
        cfg.external_id_col,
        cfg.ratio_col,
        *cfg.spatial_cols,
        *cfg.predictor_cols,
    ]
    if cfg.match_col:
        required.append(cfg.match_col)
    if cfg.stock_col and not cfg.stock_table_path:
        required.append(cfg.stock_col)
    if cfg.population_col:
        required.append(cfg.population_col)
    require_columns(raw, required, "Input analytical table")

    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    df["__external_id"] = df[cfg.external_id_col]
    df["__join_key"] = normalise_key(df[cfg.external_id_col])
    if df["__join_key"].isna().any():
        raise ValueError(f"{cfg.external_id_col} contains missing identifiers.")
    if df["__join_key"].duplicated().any():
        examples = df.loc[
            df["__join_key"].duplicated(keep=False), cfg.external_id_col
        ].head(20).tolist()
        raise ValueError(
            f"{cfg.external_id_col} must identify one row per ADM3 unit. "
            f"Duplicate examples: {examples}"
        )

    numeric_columns = [cfg.ratio_col, *cfg.spatial_cols, *cfg.predictor_cols]
    numeric_columns.extend([column for column in cfg.display_cols if column in df.columns])
    if "CII" in df.columns and "CII" not in numeric_columns:
        numeric_columns.append("CII")
    if cfg.stock_col and cfg.stock_col in df.columns:
        numeric_columns.append(cfg.stock_col)
    if cfg.population_col:
        numeric_columns.append(cfg.population_col)
    numeric_columns = list(dict.fromkeys(numeric_columns))
    df = make_numeric(df, numeric_columns)

    matched_keys = load_matched_id_keys(cfg)
    complete_core = (
        df[cfg.ratio_col].notna()
        & df[list(cfg.spatial_cols)].notna().all(axis=1)
    )
    positive_ratio = df[cfg.ratio_col] > 0

    if cfg.match_col:
        selected_mask = truthy_mask(df[cfg.match_col]) & complete_core & positive_ratio
        selection_method = (
            f"explicit record indicator {cfg.match_col}, valid coordinates and "
            f"positive {cfg.ratio_col}"
        )
    elif matched_keys is not None:
        selected_mask = (
            df["__join_key"].isin(matched_keys) & complete_core & positive_ratio
        )
        selection_method = (
            f"ID list {cfg.matched_ids_path}, valid coordinates and positive "
            f"{cfg.ratio_col}"
        )
        missing_ids = sorted(matched_keys - set(df.loc[df["__join_key"].isin(matched_keys), "__join_key"]))
        if missing_ids:
            raise ValueError(
                f"{len(missing_ids)} identifiers from the ID list were not found in "
                f"the analytical table. Examples: {missing_ids[:10]}"
            )
    elif cfg.input_is_matched_sample:
        selected_mask = complete_core & positive_ratio
        selection_method = (
            f"input declared as the analytical record set, retaining valid "
            f"positive {cfg.ratio_col} observations"
        )
    else:
        selected_mask = complete_core & positive_ratio
        selection_method = (
            f"all rows with valid projected coordinates and positive {cfg.ratio_col}"
        )

    model_df = df.loc[selected_mask].copy()
    if model_df.empty:
        raise ValueError("The analytical rule retained no observations.")

    ratios = model_df[cfg.ratio_col].to_numpy(dtype=np.float64)
    if cfg.ratio_scale == "percent":
        ratios = ratios / 100.0
    elif cfg.ratio_scale != "proportion":
        raise ValueError("ratio_scale must be 'proportion' or 'percent'.")
    model_df["__ratio_proportion"] = ratios
    model_df[cfg.target_col] = logit_transform(ratios, cfg.logit_epsilon)

    model_df = model_df.reset_index(drop=True)
    model_df[cfg.model_id_col] = np.arange(len(model_df), dtype=np.int64)
    model_df["model_row"] = np.arange(len(model_df), dtype=np.int64)

    stock_total = float("nan")
    if cfg.stock_col and cfg.stock_col in model_df.columns:
        if model_df[cfg.stock_col].isna().any():
            raise ValueError(f"{cfg.stock_col} contains missing values in the analytical sample.")
        if (model_df[cfg.stock_col] < 0).any():
            raise ValueError(f"{cfg.stock_col} contains negative values.")
        stock_total = float(model_df[cfg.stock_col].sum())
        if cfg.expected_stock_total is not None and not np.isclose(
            stock_total, cfg.expected_stock_total, rtol=0, atol=0.5
        ):
            warnings.warn(
                f"The selected stock total is {stock_total:,.0f}, not the expected "
                f"value {cfg.expected_stock_total:,.0f}. Verify the stock table.",
                RuntimeWarning,
            )

    ratio_identity_max_error = float("nan")
    if cfg.stock_col and cfg.stock_col in model_df.columns and cfg.population_col:
        if (model_df[cfg.population_col] <= 0).any():
            raise ValueError(f"{cfg.population_col} must be positive.")
        reconstructed = (
            model_df[cfg.stock_col].to_numpy(dtype=np.float64)
            / model_df[cfg.population_col].to_numpy(dtype=np.float64)
        )
        ratio_identity_max_error = float(
            np.max(np.abs(reconstructed - model_df["__ratio_proportion"].to_numpy()))
        )
        if ratio_identity_max_error > cfg.ratio_identity_tolerance:
            warnings.warn(
                f"{cfg.ratio_col} differs from {cfg.stock_col} divided by "
                f"{cfg.population_col}. Maximum absolute difference is "
                f"{ratio_identity_max_error:.6g}.",
                RuntimeWarning,
            )

    cii_shift = float("nan")
    cii_shift_range = float("nan")
    cii_additive_equivalent: Any = "not_tested"
    if "CII_pos" in model_df.columns and "CII" in model_df.columns:
        pair = model_df[["CII_pos", "CII"]].dropna()
        if not pair.empty:
            differences = (
                pair["CII_pos"].to_numpy(dtype=np.float64)
                - pair["CII"].to_numpy(dtype=np.float64)
            )
            cii_shift = float(np.mean(differences))
            cii_shift_range = float(np.ptp(differences))
            cii_additive_equivalent = bool(cii_shift_range <= 1e-10)
            if not cii_additive_equivalent:
                warnings.warn(
                    "CII_pos is not an exact additive translation of CII in the "
                    "analytical sample. Verify the variable definition.",
                    RuntimeWarning,
                )

    audit_rows: List[Dict[str, Any]] = [
        {"item": "raw_rows", "value": int(len(df))},
        {"item": "sample_selection_method", "value": selection_method},
        {"item": "model_rows", "value": int(len(model_df))},
        {"item": "unique_ADM3_units", "value": int(model_df["__join_key"].nunique())},
        {"item": "response_column", "value": cfg.ratio_col},
        {"item": "CII_model_column", "value": cfg.predictor_cols[0]},
        {"item": "CII_pos_minus_CII_mean", "value": cii_shift},
        {"item": "CII_pos_minus_CII_range", "value": cii_shift_range},
        {"item": "CII_pos_is_additive_translation", "value": cii_additive_equivalent},
        {"item": "reported_IDP_stock_total_if_available", "value": stock_total},
        {"item": "ratio_identity_max_absolute_error", "value": ratio_identity_max_error},
        {"item": "ratio_min", "value": float(model_df["__ratio_proportion"].min())},
        {"item": "ratio_max", "value": float(model_df["__ratio_proportion"].max())},
        {"item": "target_logit_min", "value": float(model_df[cfg.target_col].min())},
        {"item": "target_logit_max", "value": float(model_df[cfg.target_col].max())},
    ]
    for predictor in cfg.predictor_cols:
        audit_rows.extend(
            [
                {
                    "item": f"missing_{predictor}_within_model_sample",
                    "value": int(model_df[predictor].isna().sum()),
                },
                {
                    "item": f"unique_nonmissing_{predictor}",
                    "value": int(model_df[predictor].nunique(dropna=True)),
                },
            ]
        )
        if model_df[predictor].nunique(dropna=True) < 2:
            raise ValueError(f"Predictor {predictor} has no usable variation.")

    return model_df, pd.DataFrame(audit_rows)


def predictor_diagnostics(model_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    complete = model_df.loc[:, list(cfg.predictor_cols)].copy()
    complete = complete.fillna(complete.median(numeric_only=True))
    correlation = complete.corr(method="pearson")

    rows: List[Dict[str, Any]] = []
    values = complete.to_numpy(dtype=np.float64)
    for index, predictor in enumerate(cfg.predictor_cols):
        y = values[:, index]
        x = np.delete(values, index, axis=1)
        x = np.column_stack([x, np.ones(len(x))])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ beta
        r2 = r2_score(y, fitted)
        vif = float("inf") if r2 >= 1 - 1e-12 else float(1.0 / (1.0 - r2))
        rows.append({"Predictor": predictor, "VIF": vif, "Auxiliary_R2": float(r2)})
    return correlation, pd.DataFrame(rows)


# =============================================================================
# Spatial blocks and nested validation
# =============================================================================


def assign_blocks_to_folds(
    block_labels: np.ndarray,
    n_folds: int,
    seed: int,
) -> Dict[int, int]:
    """Assign intact blocks to folds while balancing observation counts."""

    rng = np.random.default_rng(seed)
    unique, counts = np.unique(block_labels, return_counts=True)
    block_info = list(zip(unique.astype(int).tolist(), counts.astype(int).tolist()))
    rng.shuffle(block_info)
    block_info.sort(key=lambda item: item[1], reverse=True)

    fold_sizes = np.zeros(n_folds, dtype=int)
    mapping: Dict[int, int] = {}
    for block, count in block_info:
        candidate = np.flatnonzero(fold_sizes == fold_sizes.min())
        fold = int(rng.choice(candidate))
        mapping[int(block)] = fold
        fold_sizes[fold] += int(count)
    return mapping


def create_spatial_blocks_and_folds(
    model_df: pd.DataFrame,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    coords = model_df.loc[:, list(cfg.spatial_cols)].to_numpy(dtype=np.float64)
    if len(np.unique(coords, axis=0)) < cfg.n_spatial_folds:
        raise ValueError("Too few unique coordinates for spatial cross-validation.")

    coords_scaled = StandardScaler().fit_transform(coords)
    requested_blocks = min(cfg.n_spatial_blocks, len(model_df))
    requested_blocks = max(requested_blocks, cfg.n_spatial_folds)
    unique_coord_count = len(np.unique(coords_scaled, axis=0))
    actual_blocks = min(requested_blocks, unique_coord_count)

    block_labels = KMeans(
        n_clusters=actual_blocks,
        random_state=cfg.random_seed,
        n_init=50,
    ).fit_predict(coords_scaled).astype(int)
    actual_blocks = int(len(np.unique(block_labels)))
    if actual_blocks < cfg.n_spatial_folds:
        raise RuntimeError("KMeans produced fewer spatial blocks than outer folds.")

    block_to_fold = assign_blocks_to_folds(
        block_labels,
        n_folds=cfg.n_spatial_folds,
        seed=cfg.random_seed,
    )
    fold_labels = np.array([block_to_fold[int(block)] for block in block_labels], dtype=int)
    fold_counts = np.bincount(fold_labels, minlength=cfg.n_spatial_folds)
    block_counts = np.array(
        [len(np.unique(block_labels[fold_labels == fold])) for fold in range(cfg.n_spatial_folds)]
    )
    if np.any(fold_counts < cfg.min_outer_fold_size):
        raise RuntimeError(
            "At least one outer fold is too small. "
            f"Counts are {fold_counts.tolist()}. Reduce --n-spatial-blocks or "
            "--min-outer-fold-size only with a documented justification."
        )

    metadata = {
        "requested_spatial_blocks": int(cfg.n_spatial_blocks),
        "actual_spatial_blocks": int(actual_blocks),
        "outer_fold_counts": fold_counts.astype(int).tolist(),
        "outer_fold_block_counts": block_counts.astype(int).tolist(),
    }
    return block_labels, fold_labels, metadata


def create_inner_spatial_split(
    outer_train: pd.DataFrame,
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    groups = outer_train["spatial_block"].to_numpy(dtype=int)
    if len(np.unique(groups)) < 3:
        raise RuntimeError("Too few spatial blocks remain for inner validation.")

    splitter = GroupShuffleSplit(
        n_splits=100,
        test_size=cfg.inner_validation_fraction,
        random_state=seed,
    )
    best: Optional[Tuple[np.ndarray, np.ndarray]] = None
    best_score = math.inf
    target_n = cfg.inner_validation_fraction * len(outer_train)
    minimum_train = max(30, 6 * (len(cfg.predictor_cols) + 1))
    minimum_validation = max(cfg.min_inner_validation_size, len(cfg.predictor_cols) + 3)

    for train_index, validation_index in splitter.split(outer_train, groups=groups):
        if len(train_index) < minimum_train or len(validation_index) < minimum_validation:
            continue
        score = abs(len(validation_index) - target_n)
        if score < best_score:
            best = (np.asarray(train_index), np.asarray(validation_index))
            best_score = score
    if best is None:
        raise RuntimeError("Unable to construct a valid inner spatial-block split.")
    return best


# =============================================================================
# Preprocessing, OLS and GWR
# =============================================================================


def fit_fold_preprocessor(
    train_frame: pd.DataFrame,
    predictors: Sequence[str],
    target_col: str,
) -> FoldPreprocessor:
    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(train_frame.loc[:, list(predictors)])
    x_scaler = StandardScaler().fit(x_imputed)
    if np.any(x_scaler.scale_ <= 1e-12):
        bad = [predictors[i] for i, scale in enumerate(x_scaler.scale_) if scale <= 1e-12]
        raise ValueError(f"Zero-variance training predictors: {bad}")

    y = train_frame[target_col].to_numpy(dtype=np.float64)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y, ddof=0))
    if not np.isfinite(y_std) or y_std <= 1e-12:
        raise ValueError("The training-fold response has zero or invalid variance.")
    return FoldPreprocessor(imputer=imputer, x_scaler=x_scaler, y_mean=y_mean, y_std=y_std)


def make_preprocessed_frame(
    frame: pd.DataFrame,
    preprocessor: FoldPreprocessor,
    cfg: Config,
) -> pd.DataFrame:
    out = frame.copy()
    x = preprocessor.transform_x(frame, cfg.predictor_cols)
    y = preprocessor.transform_y(frame[cfg.target_col].to_numpy(dtype=np.float64))
    for index, predictor in enumerate(cfg.predictor_cols):
        out[predictor] = x[:, index]
    out[cfg.target_col] = y
    return out


def add_intercept(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.column_stack([x, np.ones(len(x), dtype=np.float64)])


def fit_ols(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(add_intercept(x_train), y_train, rcond=None)
    return coefficients.astype(np.float64)


def predict_ols(coefficients: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    return add_intercept(x_query) @ np.asarray(coefficients, dtype=np.float64)


def adaptive_bisquare_weights(distances: np.ndarray, neighbors: int) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    n = len(distances)
    k = int(np.clip(neighbors, 2, n))
    finite = np.isfinite(distances)
    if finite.sum() < k:
        raise ValueError("Too few finite distances for GWR.")

    bandwidth = float(np.partition(distances[finite], k - 1)[k - 1])
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        positive = distances[finite & (distances > 0)]
        bandwidth = float(np.min(positive)) if len(positive) else 1.0

    scaled = distances / (bandwidth + 1e-12)
    weights = np.where((scaled < 1.0) & finite, (1.0 - scaled**2) ** 2, 0.0)
    if np.count_nonzero(weights) < k:
        nearest = np.argsort(distances)[:k]
        weights[nearest] = np.maximum(weights[nearest], 1e-10)
    return weights


def gwr_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    coords_train: np.ndarray,
    x_query: np.ndarray,
    coords_query: np.ndarray,
    neighbors: int,
    ridge: float,
) -> np.ndarray:
    x_train_aug = add_intercept(x_train)
    x_query_aug = add_intercept(x_query)
    distance_matrix = cdist(coords_query, coords_train)
    predictions = np.empty(len(x_query_aug), dtype=np.float64)

    for row in range(len(x_query_aug)):
        weights = adaptive_bisquare_weights(distance_matrix[row], neighbors)
        xtw = x_train_aug.T * weights
        xtwx = xtw @ x_train_aug
        xtwy = xtw @ np.asarray(y_train, dtype=np.float64)
        penalty = np.eye(xtwx.shape[0], dtype=np.float64) * ridge
        penalty[-1, -1] = 0.0
        try:
            beta = np.linalg.solve(xtwx + penalty, xtwy)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(xtwx + penalty) @ xtwy
        predictions[row] = float(x_query_aug[row] @ beta)
    return predictions


def gwr_candidate_neighbors(n_train: int, n_predictors: int, cfg: Config) -> List[int]:
    minimum = max(cfg.gwr_min_neighbors, n_predictors + 8)
    maximum = max(minimum, n_train - 1)
    if maximum <= minimum:
        return [maximum]
    values = np.linspace(minimum, maximum, cfg.gwr_candidate_count)
    return sorted({int(round(value)) for value in values})


def select_gwr_neighbors(
    x_train: np.ndarray,
    y_train: np.ndarray,
    coords_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    coords_validation: np.ndarray,
    cfg: Config,
) -> Tuple[int, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    for neighbors in gwr_candidate_neighbors(len(x_train), x_train.shape[1], cfg):
        prediction = gwr_predict(
            x_train,
            y_train,
            coords_train,
            x_validation,
            coords_validation,
            neighbors,
            cfg.gwr_ridge,
        )
        rows.append(
            {
                "neighbors": int(neighbors),
                "validation_RMSE": float(np.sqrt(mean_squared_error(y_validation, prediction))),
                "validation_R2": float(r2_score(y_validation, prediction)),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["validation_RMSE", "neighbors"], ascending=[True, True]
    )
    return int(result.iloc[0]["neighbors"]), result.reset_index(drop=True)


# =============================================================================
# GNNWR interface pinned to upstream 0.1.17
# =============================================================================


def init_gnnwr_split(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    cfg: Config,
) -> Tuple[Any, Any, Any]:
    datasets, _ = import_gnnwr()
    if not hasattr(datasets, "init_dataset_split"):
        raise RuntimeError(
            "The local GNNWR package lacks datasets.init_dataset_split. "
            "Use the pinned 0.1.17 revision recorded in the README."
        )

    train_dataset, validation_dataset, test_dataset = datasets.init_dataset_split(
        train_data=train_data.copy().reset_index(drop=True),
        val_data=validation_data.copy().reset_index(drop=True),
        test_data=test_data.copy().reset_index(drop=True),
        x_column=list(cfg.predictor_cols),
        y_column=[cfg.target_col],
        spatial_column=list(cfg.spatial_cols),
        id_column=[cfg.model_id_col],
        process_fn="minmax_scale",
        process_var=[],
        batch_size=max(2, len(train_data)),
        shuffle=False,
        use_model="gnnwr",
        max_val_size=max(1, len(validation_data)),
        max_test_size=max(1, len(test_data)),
        Reference="train",
        simple_distance=True,
        dropna=False,
    )

    # Explicitly preserve prediction order. This is necessary because the
    # package exposes coefficients as an array rather than an ID-indexed table.
    train_dataset.dataloader = DataLoader(
        train_dataset, batch_size=max(2, len(train_dataset)), shuffle=False
    )
    validation_dataset.dataloader = DataLoader(
        validation_dataset, batch_size=max(1, len(validation_dataset)), shuffle=False
    )
    test_dataset.dataloader = DataLoader(
        test_dataset, batch_size=max(1, len(test_dataset)), shuffle=False
    )
    return train_dataset, validation_dataset, test_dataset


def instantiate_gnnwr(
    train_dataset: Any,
    validation_dataset: Any,
    test_dataset: Any,
    cfg: Config,
    use_gpu: bool,
    model_name: str,
    model_root: Path,
) -> Any:
    _, models = import_gnnwr()
    model_save_path = model_root / "models"
    log_path = model_root / "logs"
    write_path = model_root / "runs"
    for path in (model_save_path, log_path, write_path):
        path.mkdir(parents=True, exist_ok=True)

    return models.GNNWR(
        train_dataset=train_dataset,
        valid_dataset=validation_dataset,
        test_dataset=test_dataset,
        dense_layers=list(cfg.gnnwr_dense_layers),
        start_lr=cfg.gnnwr_start_lr,
        optimizer=cfg.gnnwr_optimizer,
        drop_out=cfg.gnnwr_drop_out,
        batch_norm=cfg.gnnwr_batch_norm,
        activate_func=nn.PReLU(init=cfg.gnnwr_activation_init),
        model_name=model_name,
        model_save_path=str(model_save_path),
        write_path=str(write_path),
        use_gpu=bool(use_gpu),
        use_ols=True,
        log_path=str(log_path),
        log_file_name=f"{model_name}.log",
        optimizer_params={
            "scheduler": "Constant",
            "weight_decay": cfg.gnnwr_weight_decay,
        },
    )


def run_gnnwr_model(
    model: Any,
    max_epoch: int,
    early_stop: int,
    model_selection: str,
) -> None:
    model.run(
        max_epoch=max_epoch,
        early_stop=early_stop,
        model_selection=model_selection,
    )


def selection_best_epoch(model: Any, maximum_epoch: int) -> int:
    last_epoch = int(getattr(model, "_epoch", maximum_epoch - 1)) + 1
    no_update = int(getattr(model, "_noUpdateEpoch", 0))
    return int(np.clip(last_epoch - no_update, 1, maximum_epoch))


def gnnwr_prediction_array(model: Any, dataset: Any, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    result = model.predict(dataset).copy()
    require_columns(result, [cfg.model_id_col], "GNNWR prediction output")
    prediction_column = (
        "denormalized_pred_result"
        if "denormalized_pred_result" in result.columns
        else "pred_result"
    )
    require_columns(result, [prediction_column], "GNNWR prediction output")
    ids = pd.to_numeric(result[cfg.model_id_col], errors="raise").astype(np.int64).to_numpy()
    prediction = pd.to_numeric(result[prediction_column], errors="raise").to_numpy(dtype=np.float64)
    return ids, prediction.reshape(-1)


def train_gnnwr_for_outer_fold(
    inner_train: pd.DataFrame,
    inner_validation: pd.DataFrame,
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    cfg: Config,
    use_gpu: bool,
    fold_number: int,
    output_dir: Path,
) -> Tuple[np.ndarray, int, float]:
    """Select epoch on an inner spatial holdout and refit the outer training set."""

    fold_root = output_dir / "gnnwr_work" / f"fold_{fold_number}"
    if fold_root.exists():
        shutil.rmtree(fold_root)
    selection_root = fold_root / "selection"
    refit_root = fold_root / "refit"

    selection_preprocessor = fit_fold_preprocessor(
        inner_train, cfg.predictor_cols, cfg.target_col
    )
    inner_train_prepared = make_preprocessed_frame(inner_train, selection_preprocessor, cfg)
    inner_validation_prepared = make_preprocessed_frame(
        inner_validation, selection_preprocessor, cfg
    )
    # The outer test data satisfy the upstream constructor but are not used for
    # model selection. They remain unseen by preprocessing and early stopping.
    outer_test_selection = make_preprocessed_frame(
        outer_test, selection_preprocessor, cfg
    )
    train_ds, validation_ds, test_ds = init_gnnwr_split(
        inner_train_prepared,
        inner_validation_prepared,
        outer_test_selection,
        cfg,
    )

    set_global_seed(cfg.random_seed + 1000 + fold_number)
    selection_model = instantiate_gnnwr(
        train_ds,
        validation_ds,
        test_ds,
        cfg,
        use_gpu,
        f"GNNWR_SELECTION_FOLD_{fold_number}",
        selection_root,
    )
    run_gnnwr_model(
        selection_model,
        max_epoch=cfg.gnnwr_max_epoch,
        early_stop=cfg.gnnwr_early_stop,
        model_selection="val",
    )
    best_epoch = selection_best_epoch(selection_model, cfg.gnnwr_max_epoch)
    best_validation_r2 = tensor_or_number_to_float(
        getattr(selection_model, "_bestr2", np.nan)
    )
    close_gnnwr_writer(selection_model)
    del selection_model, train_ds, validation_ds, test_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    outer_preprocessor = fit_fold_preprocessor(
        outer_train, cfg.predictor_cols, cfg.target_col
    )
    outer_train_prepared = make_preprocessed_frame(outer_train, outer_preprocessor, cfg)
    outer_test_prepared = make_preprocessed_frame(outer_test, outer_preprocessor, cfg)
    placeholder_n = min(max(8, len(outer_train_prepared) // 10), 20)
    refit_validation = outer_train_prepared.iloc[:placeholder_n].copy()
    refit_validation[cfg.model_id_col] = -np.arange(
        1, placeholder_n + 1, dtype=np.int64
    )

    train_ds, validation_ds, test_ds = init_gnnwr_split(
        outer_train_prepared,
        refit_validation,
        outer_test_prepared,
        cfg,
    )
    set_global_seed(cfg.random_seed + 2000 + fold_number)
    refit_model = instantiate_gnnwr(
        train_ds,
        validation_ds,
        test_ds,
        cfg,
        use_gpu,
        f"GNNWR_REFIT_FOLD_{fold_number}",
        refit_root,
    )
    run_gnnwr_model(
        refit_model,
        max_epoch=best_epoch,
        early_stop=-1,
        model_selection="last",
    )
    ids, prediction_standardized = gnnwr_prediction_array(refit_model, test_ds, cfg)
    pred_frame = pd.DataFrame(
        {cfg.model_id_col: ids, "prediction_standardized": prediction_standardized}
    )
    pred_frame = outer_test[[cfg.model_id_col]].merge(
        pred_frame,
        on=cfg.model_id_col,
        how="left",
        validate="one_to_one",
    )
    if pred_frame["prediction_standardized"].isna().any():
        raise RuntimeError("GNNWR test predictions could not be aligned by model ID.")
    prediction_logit = outer_preprocessor.inverse_y(
        pred_frame["prediction_standardized"].to_numpy(dtype=np.float64)
    )

    close_gnnwr_writer(refit_model)
    del refit_model, train_ds, validation_ds, test_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction_logit, best_epoch, best_validation_r2


# =============================================================================
# Moran's I
# =============================================================================


def symmetric_knn_weights(coords: np.ndarray, k: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    if n <= k:
        raise ValueError(f"Moran k={k} requires more than {k} observations.")
    distances = cdist(coords, coords)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    adjacency = np.zeros((n, n), dtype=np.float64)
    rows = np.repeat(np.arange(n), k)
    adjacency[rows, nearest.ravel()] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    row_sums = adjacency.sum(axis=1)
    if np.any(row_sums <= 0):
        raise RuntimeError("A zero-neighbour row occurred in the Moran weight matrix.")
    return adjacency / row_sums[:, None]


def moran_i(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    centered = values - values.mean()
    denominator = float(centered @ centered)
    if denominator <= 0:
        return float("nan")
    numerator = float(centered @ weights @ centered)
    return float((len(values) / weights.sum()) * (numerator / denominator))


def global_moran_permutation(
    values: np.ndarray,
    coords: np.ndarray,
    k: int,
    permutations: int,
    seed: int,
) -> Tuple[float, float, float]:
    weights = symmetric_knn_weights(coords, k)
    observed = moran_i(values, weights)
    expected = -1.0 / (len(values) - 1)
    rng = np.random.default_rng(seed)
    simulated = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        simulated[index] = moran_i(rng.permutation(values), weights)
    p_value = (1.0 + float(np.sum(np.abs(simulated - expected) >= abs(observed - expected)))) / (
        permutations + 1.0
    )
    return observed, float(p_value), expected


# =============================================================================
# Spatial cross-validation and Table 3
# =============================================================================


def run_spatial_cross_validation(
    model_df: pd.DataFrame,
    cfg: Config,
    use_gpu: bool,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    block_labels, fold_labels, partition_metadata = create_spatial_blocks_and_folds(
        model_df, cfg
    )
    work_df = model_df.copy()
    work_df["spatial_block"] = block_labels
    work_df["outer_fold"] = fold_labels

    assignment_columns = [
        cfg.model_id_col,
        "__external_id",
        "__join_key",
        "model_row",
        "spatial_block",
        "outer_fold",
        *cfg.spatial_cols,
    ]
    fold_assignments = work_df[assignment_columns].copy()

    prediction_rows: List[pd.DataFrame] = []
    fold_metric_rows: List[Dict[str, Any]] = []
    gwr_selection_rows: List[pd.DataFrame] = []

    for fold in range(cfg.n_spatial_folds):
        fold_number = fold + 1
        print("\n" + "=" * 88)
        print(f"OUTER SPATIAL FOLD {fold_number}/{cfg.n_spatial_folds}")
        print("=" * 88)

        outer_test = work_df.loc[work_df["outer_fold"] == fold].copy().reset_index(drop=True)
        outer_train = work_df.loc[work_df["outer_fold"] != fold].copy().reset_index(drop=True)
        inner_train_index, inner_validation_index = create_inner_spatial_split(
            outer_train,
            cfg,
            seed=cfg.random_seed + fold_number,
        )
        inner_train = outer_train.iloc[inner_train_index].copy().reset_index(drop=True)
        inner_validation = outer_train.iloc[inner_validation_index].copy().reset_index(drop=True)

        print(
            f"outer train={len(outer_train)}, inner train={len(inner_train)}, "
            f"inner validation={len(inner_validation)}, outer test={len(outer_test)}"
        )

        inner_preprocessor = fit_fold_preprocessor(
            inner_train, cfg.predictor_cols, cfg.target_col
        )
        x_inner_train = inner_preprocessor.transform_x(inner_train, cfg.predictor_cols)
        x_inner_validation = inner_preprocessor.transform_x(
            inner_validation, cfg.predictor_cols
        )
        y_inner_train = inner_preprocessor.transform_y(
            inner_train[cfg.target_col].to_numpy(dtype=np.float64)
        )
        y_inner_validation = inner_preprocessor.transform_y(
            inner_validation[cfg.target_col].to_numpy(dtype=np.float64)
        )
        coords_inner_train = inner_train.loc[:, list(cfg.spatial_cols)].to_numpy(dtype=np.float64)
        coords_inner_validation = inner_validation.loc[:, list(cfg.spatial_cols)].to_numpy(dtype=np.float64)
        selected_neighbors, gwr_selection = select_gwr_neighbors(
            x_inner_train,
            y_inner_train,
            coords_inner_train,
            x_inner_validation,
            y_inner_validation,
            coords_inner_validation,
            cfg,
        )
        gwr_selection.insert(0, "outer_fold", fold_number)
        gwr_selection_rows.append(gwr_selection)
        print(f"GWR selected adaptive neighbour count={selected_neighbors}")

        outer_preprocessor = fit_fold_preprocessor(
            outer_train, cfg.predictor_cols, cfg.target_col
        )
        x_outer_train = outer_preprocessor.transform_x(outer_train, cfg.predictor_cols)
        x_outer_test = outer_preprocessor.transform_x(outer_test, cfg.predictor_cols)
        y_outer_train_standard = outer_preprocessor.transform_y(
            outer_train[cfg.target_col].to_numpy(dtype=np.float64)
        )
        y_outer_test_logit = outer_test[cfg.target_col].to_numpy(dtype=np.float64)
        coords_outer_train = outer_train.loc[:, list(cfg.spatial_cols)].to_numpy(dtype=np.float64)
        coords_outer_test = outer_test.loc[:, list(cfg.spatial_cols)].to_numpy(dtype=np.float64)

        ols_beta = fit_ols(x_outer_train, y_outer_train_standard)
        ols_pred_logit = outer_preprocessor.inverse_y(
            predict_ols(ols_beta, x_outer_test)
        )

        final_neighbors = int(min(selected_neighbors, len(outer_train) - 1))
        gwr_pred_standard = gwr_predict(
            x_outer_train,
            y_outer_train_standard,
            coords_outer_train,
            x_outer_test,
            coords_outer_test,
            final_neighbors,
            cfg.gwr_ridge,
        )
        gwr_pred_logit = outer_preprocessor.inverse_y(gwr_pred_standard)

        gnnwr_pred_logit, selected_epoch, inner_best_r2 = train_gnnwr_for_outer_fold(
            inner_train,
            inner_validation,
            outer_train,
            outer_test,
            cfg,
            use_gpu,
            fold_number,
            output_dir,
        )
        print(
            f"GNNWR selected epoch={selected_epoch}; inner validation best "
            f"R²={inner_best_r2:.4f}"
        )

        model_predictions = {
            "GNNWR": gnnwr_pred_logit,
            "GWR": gwr_pred_logit,
            "OLS": ols_pred_logit,
        }
        for model_name, prediction in model_predictions.items():
            prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
            if len(prediction) != len(outer_test) or np.any(~np.isfinite(prediction)):
                raise RuntimeError(
                    f"{model_name} fold {fold_number} produced invalid predictions."
                )
            metrics = calculate_metrics(y_outer_test_logit, prediction)
            print(
                f"{model_name:6s} R²={metrics['R2']:.4f}, "
                f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}"
            )
            fold_metric_rows.append(
                {
                    "outer_fold": fold_number,
                    "Model": model_name,
                    "n_outer_train": int(len(outer_train)),
                    "n_outer_test": int(len(outer_test)),
                    "R2": metrics["R2"],
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "Bias": metrics["Bias"],
                    "GNNWR_selected_epoch": int(selected_epoch),
                    "GNNWR_inner_best_R2": float(inner_best_r2),
                    "GWR_selected_neighbors": int(final_neighbors),
                }
            )

            current = pd.DataFrame(
                {
                    cfg.model_id_col: outer_test[cfg.model_id_col].to_numpy(dtype=np.int64),
                    "external_id": outer_test["__external_id"].to_numpy(),
                    "join_key": outer_test["__join_key"].to_numpy(),
                    "model_row": outer_test["model_row"].to_numpy(dtype=np.int64),
                    "spatial_block": outer_test["spatial_block"].to_numpy(dtype=int),
                    "outer_fold": int(fold_number),
                    "Model": model_name,
                    "y_true_logit": y_outer_test_logit,
                    "y_pred_logit": prediction,
                    "residual_logit": y_outer_test_logit - prediction,
                    "y_true_ratio": expit(y_outer_test_logit),
                    "y_pred_ratio": expit(prediction),
                    cfg.spatial_cols[0]: coords_outer_test[:, 0],
                    cfg.spatial_cols[1]: coords_outer_test[:, 1],
                }
            )
            prediction_rows.append(current)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    gwr_selections = pd.concat(gwr_selection_rows, ignore_index=True)

    expected_ids = set(work_df[cfg.model_id_col].tolist())
    for model_name in ("GNNWR", "GWR", "OLS"):
        subset = predictions.loc[predictions["Model"] == model_name]
        if len(subset) != len(work_df):
            raise RuntimeError(
                f"{model_name} has {len(subset)} OOF predictions, expected {len(work_df)}."
            )
        if subset[cfg.model_id_col].duplicated().any():
            raise RuntimeError(f"{model_name} has duplicate OOF model IDs.")
        if set(subset[cfg.model_id_col].tolist()) != expected_ids:
            raise RuntimeError(f"{model_name} OOF IDs do not match the analytical sample.")

    return predictions, fold_metrics, fold_assignments, gwr_selections, partition_metadata


def build_table3(
    predictions: pd.DataFrame,
    cfg: Config,
    actual_spatial_blocks: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logit_rows: List[Dict[str, Any]] = []
    ratio_rows: List[Dict[str, Any]] = []

    for model_name in ("GNNWR", "GWR", "OLS"):
        subset = predictions.loc[predictions["Model"] == model_name].sort_values("model_row")
        y_true = subset["y_true_logit"].to_numpy(dtype=np.float64)
        y_pred = subset["y_pred_logit"].to_numpy(dtype=np.float64)
        residual = subset["residual_logit"].to_numpy(dtype=np.float64)
        coords = subset.loc[:, list(cfg.spatial_cols)].to_numpy(dtype=np.float64)
        metrics = calculate_metrics(y_true, y_pred)
        moran_value, moran_p, expected_i = global_moran_permutation(
            residual,
            coords,
            cfg.moran_k,
            cfg.moran_permutations,
            cfg.random_seed,
        )
        logit_rows.append(
            {
                "Model": model_name,
                "Validation": (
                    f"{cfg.n_spatial_folds}-fold spatial-block cross-validation "
                    f"using {actual_spatial_blocks} spatial blocks"
                ),
                "n": int(len(subset)),
                "Scale": "logit-transformed origin-hromada IDP stock ratio",
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "Bias": metrics["Bias"],
                "Residual_Moran_I": moran_value,
                "Moran_expected_I": expected_i,
                "Moran_p_permutation": moran_p,
                "Moran_k": int(cfg.moran_k),
                "Moran_permutations": int(cfg.moran_permutations),
            }
        )
        ratio_metrics = calculate_metrics(
            subset["y_true_ratio"].to_numpy(dtype=np.float64),
            subset["y_pred_ratio"].to_numpy(dtype=np.float64),
        )
        ratio_rows.append({"Model": model_name, "n": int(len(subset)), **ratio_metrics})

    detailed = pd.DataFrame(logit_rows)
    rounded = pd.DataFrame(
        {
            "Model": detailed["Model"],
            "Spatial-CV R²": detailed["R2"].map(lambda value: f"{value:.3f}"),
            "RMSE": detailed["RMSE"].map(lambda value: f"{value:.3f}"),
            "MAE": detailed["MAE"].map(lambda value: f"{value:.3f}"),
            "Residual Moran's I": detailed["Residual_Moran_I"].map(
                lambda value: f"{value:.3f}"
            ),
            "p value": detailed["Moran_p_permutation"].map(format_p_value),
        }
    )
    return detailed, rounded, pd.DataFrame(ratio_rows)


# =============================================================================
# Final GNNWR and coefficient extraction
# =============================================================================


def back_transform_local_coefficients(
    coefficient_standardized: np.ndarray,
    preprocessor: FoldPreprocessor,
) -> np.ndarray:
    """Convert standardized local slopes to logit-response slopes in raw X units."""

    coefficient_standardized = np.asarray(coefficient_standardized, dtype=np.float64)
    p = coefficient_standardized.shape[1] - 1
    if p != len(preprocessor.x_scaler.mean_):
        raise ValueError("Coefficient dimension does not match the fitted scaler.")

    slopes_standardized = coefficient_standardized[:, :p]
    intercept_standardized = coefficient_standardized[:, p]
    slopes_raw = (
        preprocessor.y_std
        * slopes_standardized
        / preprocessor.x_scaler.scale_[None, :]
    )
    intercept_raw = (
        preprocessor.y_mean
        + preprocessor.y_std * intercept_standardized
        - np.sum(slopes_raw * preprocessor.x_scaler.mean_[None, :], axis=1)
    )
    return np.column_stack([slopes_raw, intercept_raw])


def fit_final_gnnwr_seed(
    model_df: pd.DataFrame,
    cfg: Config,
    use_gpu: bool,
    selected_epoch: int,
    seed: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    preprocessor = fit_fold_preprocessor(model_df, cfg.predictor_cols, cfg.target_col)
    prepared = make_preprocessed_frame(model_df, preprocessor, cfg)

    placeholder_n = min(max(8, len(prepared) // 10), 20)
    validation = prepared.iloc[:placeholder_n].copy()
    validation[cfg.model_id_col] = -np.arange(1, placeholder_n + 1, dtype=np.int64)
    # The complete analytical sample is also supplied as the prediction dataset.
    # It is not used for epoch selection. The fixed epoch was selected by nested
    # spatial validation before this final full-sample refit.
    train_ds, validation_ds, prediction_ds = init_gnnwr_split(
        prepared,
        validation,
        prepared,
        cfg,
    )

    model_root = output_dir / "gnnwr_work" / "final" / f"seed_{seed}"
    if model_root.exists():
        shutil.rmtree(model_root)
    set_global_seed(seed)
    model = instantiate_gnnwr(
        train_ds,
        validation_ds,
        prediction_ds,
        cfg,
        use_gpu,
        f"GNNWR_FINAL_SEED_{seed}",
        model_root,
    )
    run_gnnwr_model(
        model,
        max_epoch=selected_epoch,
        early_stop=-1,
        model_selection="last",
    )
    if not hasattr(model, "predict_coef"):
        raise RuntimeError(
            "The local GNNWR package lacks predict_coef. Use the pinned upstream revision."
        )
    coefficients_standardized = np.asarray(
        model.predict_coef(prediction_ds), dtype=np.float64
    )
    if coefficients_standardized.shape != (
        len(model_df),
        len(cfg.predictor_cols) + 1,
    ):
        raise RuntimeError(
            "Unexpected local-coefficient shape: "
            f"{coefficients_standardized.shape}."
        )

    ids, predictions_standardized = gnnwr_prediction_array(model, prediction_ds, cfg)
    coefficient_ids = np.asarray(prediction_ds.id_data).reshape(-1).astype(np.int64)
    if not np.array_equal(ids, coefficient_ids):
        raise RuntimeError("GNNWR prediction and coefficient ID orders differ.")

    x_standardized = np.asarray(prediction_ds.x_data, dtype=np.float64)[:, :-1]
    prediction_from_coefficients = (
        np.sum(coefficients_standardized[:, :-1] * x_standardized, axis=1)
        + coefficients_standardized[:, -1]
    )
    max_standardized_identity_error = float(
        np.max(np.abs(prediction_from_coefficients - predictions_standardized))
    )
    if max_standardized_identity_error > 1e-4:
        raise RuntimeError(
            "The extracted standardized coefficients do not reconstruct GNNWR "
            f"predictions. Maximum error is {max_standardized_identity_error:.6g}."
        )

    coefficients_raw = back_transform_local_coefficients(
        coefficients_standardized, preprocessor
    )
    x_raw_imputed = preprocessor.impute_x(model_df, cfg.predictor_cols)
    prediction_raw_identity = (
        np.sum(coefficients_raw[:, :-1] * x_raw_imputed, axis=1)
        + coefficients_raw[:, -1]
    )
    predictions_logit = preprocessor.inverse_y(predictions_standardized)
    max_raw_identity_error = float(
        np.max(np.abs(prediction_raw_identity - predictions_logit))
    )
    if max_raw_identity_error > 1e-4:
        raise RuntimeError(
            "Back-transformed coefficients do not reconstruct logit predictions. "
            f"Maximum error is {max_raw_identity_error:.6g}."
        )

    id_frame = pd.DataFrame(
        {
            cfg.model_id_col: coefficient_ids,
            "prediction_standardized": predictions_standardized,
            "prediction_logit": predictions_logit,
            "prediction_ratio": expit(predictions_logit),
        }
    )
    coefficient_frame = pd.DataFrame({cfg.model_id_col: coefficient_ids})
    for index, predictor in enumerate(cfg.predictor_cols):
        coefficient_frame[f"coef_std_{predictor}"] = coefficients_standardized[:, index]
        coefficient_frame[f"coef_raw_{predictor}"] = coefficients_raw[:, index]
    coefficient_frame["intercept_std"] = coefficients_standardized[:, -1]
    coefficient_frame["intercept_raw_logit"] = coefficients_raw[:, -1]

    output = (
        model_df.merge(
            id_frame,
            on=cfg.model_id_col,
            how="left",
            validate="one_to_one",
        )
        .merge(
            coefficient_frame,
            on=cfg.model_id_col,
            how="left",
            validate="one_to_one",
        )
        .sort_values("model_row")
        .reset_index(drop=True)
    )
    output["final_seed"] = seed
    output["final_epoch"] = selected_epoch

    trainable_parameters = int(
        sum(parameter.numel() for parameter in model._model.parameters() if parameter.requires_grad)
    )
    metadata = {
        "seed": int(seed),
        "selected_epoch": int(selected_epoch),
        "trainable_neural_parameters": trainable_parameters,
        "observations": int(len(model_df)),
        "parameters_per_observation": float(trainable_parameters / len(model_df)),
        "coefficient_prediction_identity_error_standardized": max_standardized_identity_error,
        "coefficient_prediction_identity_error_raw": max_raw_identity_error,
        "in_sample_logit_metrics": calculate_metrics(
            model_df[cfg.target_col].to_numpy(dtype=np.float64), predictions_logit
        ),
    }

    close_gnnwr_writer(model)
    del model, train_ds, validation_ds, prediction_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output, metadata


def fit_final_gnnwr_and_extract_coefficients(
    model_df: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    cfg: Config,
    use_gpu: bool,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    epoch_rows = (
        fold_metrics.loc[fold_metrics["Model"] == "GNNWR"]
        .sort_values("outer_fold")
        .drop_duplicates(subset=["outer_fold"], keep="first")
    )
    if len(epoch_rows) != cfg.n_spatial_folds:
        raise RuntimeError(
            "Could not recover one selected GNNWR epoch for each outer fold."
        )
    epochs = epoch_rows["GNNWR_selected_epoch"].astype(int).to_numpy()
    selected_epoch = int(np.median(epochs))
    selected_epoch = max(1, selected_epoch)

    seed_outputs: List[pd.DataFrame] = []
    seed_metadata: List[Dict[str, Any]] = []
    for seed in cfg.final_seeds:
        print(f"\nFitting final full-sample GNNWR with seed {seed} at epoch {selected_epoch}")
        output, metadata = fit_final_gnnwr_seed(
            model_df,
            cfg,
            use_gpu,
            selected_epoch,
            seed,
            output_dir,
        )
        seed_outputs.append(output)
        seed_metadata.append(metadata)

    coefficient_columns = [f"coef_std_{name}" for name in cfg.predictor_cols]
    all_seed_rows: List[pd.DataFrame] = []
    for frame, seed in zip(seed_outputs, cfg.final_seeds):
        selected = frame[
            [cfg.model_id_col, "__external_id", "__join_key", *coefficient_columns]
        ].copy()
        selected.insert(3, "seed", seed)
        all_seed_rows.append(selected)
    all_seeds = pd.concat(all_seed_rows, ignore_index=True)

    primary = seed_outputs[0].copy()
    stability_rows: List[Dict[str, Any]] = []
    aggregate = model_df[[cfg.model_id_col, "__external_id", "__join_key"]].copy()
    for predictor, column in zip(cfg.predictor_cols, coefficient_columns):
        pivot = all_seeds.pivot(index=cfg.model_id_col, columns="seed", values=column)
        median = pivot.median(axis=1)
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1, ddof=0)
        sign_stability = pivot.apply(
            lambda row: float(np.mean(np.sign(row.to_numpy()) == np.sign(np.median(row.to_numpy())))),
            axis=1,
        )
        aggregate = aggregate.merge(
            pd.DataFrame(
                {
                    cfg.model_id_col: pivot.index.astype(int),
                    f"coef_median_{predictor}": median.to_numpy(),
                    f"coef_mean_{predictor}": mean.to_numpy(),
                    f"coef_sd_{predictor}": std.to_numpy(),
                    f"sign_stability_{predictor}": sign_stability.to_numpy(),
                }
            ),
            on=cfg.model_id_col,
            how="left",
            validate="one_to_one",
        )
        stability_rows.append(
            {
                "Predictor": predictor,
                "n_seeds": len(cfg.final_seeds),
                "mean_across_location_SD": float(std.mean()),
                "median_sign_stability": float(sign_stability.median()),
                "minimum_sign_stability": float(sign_stability.min()),
            }
        )

    if cfg.map_coefficient_estimator == "median" and len(cfg.final_seeds) > 1:
        for predictor in cfg.predictor_cols:
            primary[f"coef_map_{predictor}"] = primary[cfg.model_id_col].map(
                aggregate.set_index(cfg.model_id_col)[f"coef_median_{predictor}"]
            )
    else:
        for predictor in cfg.predictor_cols:
            primary[f"coef_map_{predictor}"] = primary[f"coef_std_{predictor}"]

    metadata = {
        "final_epoch_rule": "median of inner-selected epochs across the five outer folds",
        "final_epoch": selected_epoch,
        "outer_selected_epochs": sorted(int(value) for value in epochs),
        "map_coefficient_estimator": cfg.map_coefficient_estimator,
        "seed_runs": seed_metadata,
    }
    return primary, all_seeds, pd.DataFrame(stability_rows), metadata


# =============================================================================
# Fig. 6
# =============================================================================


def attach_optional_stock_table(
    coefficients: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Attach raw 2024 IDP stock counts for Fig. 6a when supplied separately."""

    if not cfg.stock_table_path:
        return coefficients
    if not cfg.stock_col:
        raise ValueError("--stock-col is required when --stock-table is supplied.")

    path = Path(cfg.stock_table_path).expanduser().resolve()
    table = safe_read_table(path)
    table.columns = table.columns.astype(str).str.strip()
    id_column = cfg.stock_table_id_col or cfg.external_id_col
    require_columns(table, [id_column, cfg.stock_col], path.name)
    table = table[[id_column, cfg.stock_col]].copy()
    table["__join_key"] = normalise_key(table[id_column])
    table[cfg.stock_col] = pd.to_numeric(table[cfg.stock_col], errors="coerce")
    table = table.dropna(subset=["__join_key", cfg.stock_col])
    if (table[cfg.stock_col] < 0).any():
        raise ValueError(f"{cfg.stock_col} contains negative values in {path.name}.")
    if table["__join_key"].duplicated().any():
        examples = table.loc[
            table["__join_key"].duplicated(keep=False), id_column
        ].head(20).tolist()
        raise ValueError(
            "The stock table must contain one record per ADM3 unit. "
            f"Duplicate examples: {examples}"
        )

    out = coefficients.drop(columns=[cfg.stock_col], errors="ignore").merge(
        table[["__join_key", cfg.stock_col]],
        on="__join_key",
        how="left",
        validate="one_to_one",
    )
    return out


def import_mapping_dependencies() -> Tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, Normalize, TwoSlopeNorm
        from matplotlib.patches import Patch
        from matplotlib.cm import ScalarMappable
    except ImportError as exc:
        raise ImportError(
            "Fig. 6 requires geopandas, shapely and matplotlib. Install the "
            "companion requirements file."
        ) from exc
    return gpd, plt, BoundaryNorm, Normalize, TwoSlopeNorm, (Patch, ScalarMappable)


def read_boundary(cfg: Config) -> Any:
    gpd, *_ = import_mapping_dependencies()
    if not cfg.boundary_path:
        raise ValueError("--boundary is required unless --skip-figure is used.")
    path = Path(cfg.boundary_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Boundary file not found: {path}")
    boundary = gpd.read_file(path, layer=cfg.boundary_layer)
    if boundary.empty:
        raise ValueError("The boundary layer is empty.")
    if boundary.crs is None:
        raise ValueError("The boundary layer has no coordinate reference system.")
    boundary_id_col = cfg.boundary_id_col or cfg.external_id_col
    require_columns(boundary, [boundary_id_col], "Boundary layer")
    boundary = boundary.copy()
    boundary["__join_key"] = normalise_key(boundary[boundary_id_col])
    boundary = boundary.loc[boundary["__join_key"].notna()].copy()
    if boundary["__join_key"].duplicated().any():
        aggregation: Dict[str, Any] = {}
        for column in boundary.columns:
            if column not in {"geometry", "__join_key"}:
                aggregation[column] = "first"
        boundary = boundary.dissolve(by="__join_key", aggfunc=aggregation, as_index=False)
    try:
        boundary["geometry"] = boundary.geometry.make_valid()
    except Exception:
        boundary["geometry"] = boundary.buffer(0)
    boundary = boundary.to_crs(cfg.map_crs)
    return boundary


def join_model_to_boundary(boundary: Any, coefficients: pd.DataFrame, cfg: Config) -> Any:
    selected_columns = [
        "__join_key",
        "__external_id",
        "__ratio_proportion",
        *[f"coef_map_{name}" for name in cfg.predictor_cols],
    ]
    if cfg.stock_col and cfg.stock_col in coefficients.columns:
        selected_columns.append(cfg.stock_col)
    selected_columns = list(dict.fromkeys(selected_columns))
    data = coefficients[selected_columns].copy()
    if data["__join_key"].duplicated().any():
        raise ValueError("The model coefficient table has duplicate hromada join keys.")
    joined = boundary.merge(data, on="__join_key", how="left", validate="one_to_one")
    missing = sorted(set(data["__join_key"]) - set(joined.loc[joined["__external_id"].notna(), "__join_key"]))
    if missing:
        raise ValueError(
            f"{len(missing)} modelled ADM3 units did not join to the boundary layer. "
            f"Examples: {missing[:10]}"
        )
    return joined


def add_north_arrow(ax: Any) -> None:
    ax.annotate(
        "N",
        xy=(0.94, 0.92),
        xytext=(0.94, 0.79),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        arrowprops={"facecolor": "black", "width": 2.2, "headwidth": 8},
    )


def add_scale_bar(ax: Any, length_km: float = 200.0) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.07 * (xmax - xmin)
    y0 = ymin + 0.06 * (ymax - ymin)
    length = length_km * 1000.0
    if length > 0.35 * (xmax - xmin):
        length = 0.25 * (xmax - xmin)
        length_km = round(length / 1000.0)
    ax.plot([x0, x0 + length], [y0, y0], color="black", linewidth=2.0)
    ax.plot([x0, x0], [y0 - 0.007 * (ymax - ymin), y0 + 0.007 * (ymax - ymin)], color="black", linewidth=1.5)
    ax.plot([x0 + length, x0 + length], [y0 - 0.007 * (ymax - ymin), y0 + 0.007 * (ymax - ymin)], color="black", linewidth=1.5)
    ax.text(x0 + length / 2, y0 + 0.015 * (ymax - ymin), f"{length_km:g} km", ha="center", va="bottom", fontsize=8)


def plot_occupied_overlay(ax: Any, cfg: Config, target_crs: Any) -> bool:
    if not cfg.occupied_path:
        return False
    gpd, *_ = import_mapping_dependencies()
    path = Path(cfg.occupied_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Occupied-area layer not found: {path}")
    occupied = gpd.read_file(path, layer=cfg.occupied_layer)
    if occupied.crs is None:
        raise ValueError("The occupied-area layer has no CRS.")
    occupied = occupied.to_crs(target_crs)
    occupied.plot(
        ax=ax,
        facecolor="none",
        edgecolor="0.35",
        linewidth=0.35,
        hatch="////",
        zorder=5,
    )
    return True


def create_figure6(
    coefficients: pd.DataFrame,
    cfg: Config,
    output_dir: Path,
) -> Tuple[List[Path], Any]:
    """Create Fig. 6 without assigning coefficients to unmodelled ADM3 units."""

    gpd, plt, BoundaryNorm, Normalize, TwoSlopeNorm, patch_classes = import_mapping_dependencies()
    Patch, ScalarMappable = patch_classes
    boundary = read_boundary(cfg)
    joined = join_model_to_boundary(boundary, coefficients, cfg)

    fig = plt.figure(figsize=(15.0, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1])
    ax_a = fig.add_subplot(grid[:, 0])
    axes = [
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]

    for axis in [ax_a, *axes]:
        joined.plot(ax=axis, facecolor="white", edgecolor="0.75", linewidth=0.18, zorder=1)
        axis.set_axis_off()
        axis.set_aspect("equal")

    # Panel a uses raw stock counts when provided. The analytical workbook does
    # not contain a raw stock-count field, so the stock ratio is used otherwise.
    stock_available = bool(
        cfg.stock_col
        and cfg.stock_col in joined.columns
        and joined[cfg.stock_col].notna().any()
    )
    panel_a_legend_handles = None
    panel_a_legend_title = None
    if stock_available:
        stock_bins = np.array([0, 50, 200, 1000, 5000, np.inf], dtype=float)
        stock_labels = ["1–49", "50–199", "200–999", "1,000–4,999", "≥5,000"]
        stock_colors = plt.get_cmap("YlOrRd", len(stock_labels))(np.arange(len(stock_labels)))
        stock_values = joined[cfg.stock_col]
        stock_class = pd.cut(
            stock_values,
            bins=stock_bins,
            labels=False,
            include_lowest=True,
            right=False,
        )
        for class_index, color in enumerate(stock_colors):
            subset = joined.loc[stock_class == class_index]
            if not subset.empty:
                subset.plot(
                    ax=ax_a,
                    facecolor=color,
                    edgecolor="0.45",
                    linewidth=0.20,
                    zorder=2,
                )
        panel_a_legend_handles = [
            Patch(facecolor=color, edgecolor="0.45", label=label)
            for color, label in zip(stock_colors, stock_labels)
        ]
        panel_a_legend_handles.append(
            Patch(facecolor="white", edgecolor="0.65", label="No stock record in the supplied table")
        )
        panel_a_title = "2024 origin hromada IDP stock"
        panel_a_legend_title = "Reported IDP stock"
    else:
        ratio_column = "__ratio_proportion"
        ratio_values = joined[ratio_column].dropna().to_numpy(dtype=np.float64)
        if not len(ratio_values):
            raise ValueError("No response values joined to the ADM3 boundary for Fig. 6a.")
        positive = ratio_values[ratio_values > 0]
        if not len(positive):
            raise ValueError("No positive stock-ratio values are available for Fig. 6a.")
        vmin = float(np.min(positive))
        vmax = float(np.max(positive))
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("YlOrRd")
        mapped_ratio = joined.loc[joined[ratio_column].notna()]
        mapped_ratio.plot(
            ax=ax_a,
            column=ratio_column,
            cmap=cmap,
            norm=norm,
            edgecolor="0.45",
            linewidth=0.20,
            zorder=2,
        )
        scalar = ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        colorbar = fig.colorbar(
            scalar,
            ax=ax_a,
            orientation="horizontal",
            fraction=0.035,
            pad=0.015,
            aspect=32,
        )
        colorbar.ax.tick_params(labelsize=7, length=2)
        colorbar.set_label("Reported origin hromada IDP stock ratio", fontsize=8)
        panel_a_title = "2024 origin hromada IDP stock ratio"

    occupied_drawn = plot_occupied_overlay(ax_a, cfg, joined.crs)
    if panel_a_legend_handles is not None:
        if occupied_drawn:
            panel_a_legend_handles.append(
                Patch(facecolor="white", edgecolor="0.35", hatch="////", label="Occupied area")
            )
        ax_a.legend(
            handles=panel_a_legend_handles,
            title=panel_a_legend_title,
            loc="lower left",
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )
    elif occupied_drawn:
        ax_a.legend(
            handles=[Patch(facecolor="white", edgecolor="0.35", hatch="////", label="Occupied area")],
            loc="lower left",
            frameon=False,
            fontsize=7,
        )
    ax_a.set_title(panel_a_title, fontsize=11, pad=5)
    ax_a.text(0.01, 0.985, "a", transform=ax_a.transAxes, ha="left", va="top", fontsize=13, fontweight="bold")
    add_north_arrow(ax_a)
    add_scale_bar(ax_a)

    # Manuscript panel order is older adult share, CII, BD and NTL extinction.
    panel_specs = [
        (cfg.predictor_cols[1], cfg.coefficient_labels[1], "b"),
        (cfg.predictor_cols[0], cfg.coefficient_labels[0], "c"),
        (cfg.predictor_cols[2], cfg.coefficient_labels[2], "d"),
        (cfg.predictor_cols[3], cfg.coefficient_labels[3], "e"),
    ]
    for axis, (predictor, title, panel_label) in zip(axes, panel_specs):
        column = f"coef_map_{predictor}"
        values = joined[column].dropna().to_numpy(dtype=np.float64)
        if not len(values):
            raise ValueError(f"No mapped coefficient values are available for {predictor}.")
        value_min = float(np.min(values))
        value_max = float(np.max(values))
        if value_min < 0 < value_max:
            limit = max(abs(value_min), abs(value_max))
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
            cmap = plt.get_cmap("RdBu_r")
        else:
            if np.isclose(value_min, value_max):
                value_min -= 0.5
                value_max += 0.5
            norm = Normalize(vmin=value_min, vmax=value_max)
            cmap = plt.get_cmap("viridis")
        mapped_coefficients = joined.loc[joined[column].notna()]
        mapped_coefficients.plot(
            ax=axis,
            column=column,
            cmap=cmap,
            norm=norm,
            edgecolor="0.45",
            linewidth=0.20,
            zorder=2,
        )
        coefficient_occupied = plot_occupied_overlay(axis, cfg, joined.crs)
        axis.set_title(title, fontsize=10, pad=4)
        axis.text(0.01, 0.985, panel_label, transform=axis.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
        add_north_arrow(axis)
        add_scale_bar(axis, length_km=200.0)
        if coefficient_occupied:
            axis.legend(
                handles=[Patch(facecolor="white", edgecolor="0.35", hatch="////", label="Occupied area")],
                loc="lower left",
                frameon=False,
                fontsize=7,
            )
        scalar = ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        colorbar = fig.colorbar(
            scalar,
            ax=axis,
            orientation="horizontal",
            fraction=0.045,
            pad=0.015,
            aspect=28,
        )
        colorbar.ax.tick_params(labelsize=7, length=2)
        colorbar.set_label("Standardized local coefficient", fontsize=8)

    output_paths = [
        output_dir / "Figure6_section3_4.png",
        output_dir / "Figure6_section3_4.pdf",
        output_dir / "Figure6_section3_4.tif",
    ]
    fig.savefig(output_paths[0], dpi=cfg.figure_dpi, bbox_inches="tight")
    fig.savefig(output_paths[1], bbox_inches="tight")
    try:
        fig.savefig(
            output_paths[2],
            dpi=cfg.figure_dpi,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except TypeError:
        fig.savefig(output_paths[2], dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return output_paths, joined


# =============================================================================
# Results summary and manifest
# =============================================================================


def coefficient_summary(coefficients: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for predictor, label in zip(cfg.predictor_cols, cfg.coefficient_labels):
        values = coefficients[f"coef_map_{predictor}"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "Predictor": predictor,
                "Label": label,
                "Minimum": float(np.min(values)),
                "Q25": float(np.quantile(values, 0.25)),
                "Median": float(np.median(values)),
                "Q75": float(np.quantile(values, 0.75)),
                "Maximum": float(np.max(values)),
                "Positive_share": float(np.mean(values > 0)),
                "Negative_share": float(np.mean(values < 0)),
            }
        )
    return pd.DataFrame(rows)


def write_section34_summary(
    model_df: pd.DataFrame,
    table3: pd.DataFrame,
    coefficient_stats: pd.DataFrame,
    final_metadata: Mapping[str, Any],
    cfg: Config,
    output_path: Path,
    mapped: Optional[Any] = None,
) -> None:
    lines: List[str] = []
    lines.append("Section 3.4 reproducibility summary")
    lines.append("=" * 42)
    lines.append(f"Analytical sample: {len(model_df)} ADM3 observations retained by the declared rule")
    if cfg.stock_col and cfg.stock_col in model_df.columns:
        lines.append(
            f"Reported IDP stock: {model_df[cfg.stock_col].sum():,.0f} persons"
        )
    lines.append("Response scale for Table 3: logit-transformed origin-hromada stock ratio")
    lines.append("")
    lines.append("Spatially out-of-fold model performance")
    for _, row in table3.iterrows():
        lines.append(
            f"{row['Model']}: R2={row['R2']:.3f}, RMSE={row['RMSE']:.3f}, "
            f"MAE={row['MAE']:.3f}, residual Moran I={row['Residual_Moran_I']:.3f}, "
            f"p={row['Moran_p_permutation']:.3f}"
        )
    lines.append("")
    lines.append(
        f"Final GNNWR epoch: {final_metadata['final_epoch']} selected as the median "
        "of the five inner spatial-validation epochs"
    )
    lines.append(
        f"Coefficient map estimator: {final_metadata['map_coefficient_estimator']}"
    )
    lines.append("")
    lines.append("Local coefficient distributions on the standardized scale")
    for _, row in coefficient_stats.iterrows():
        lines.append(
            f"{row['Predictor']}: min={row['Minimum']:.4f}, median={row['Median']:.4f}, "
            f"max={row['Maximum']:.4f}, positive share={row['Positive_share']:.3f}"
        )

    if mapped is not None and cfg.oblast_col and cfg.oblast_col in mapped.columns:
        lines.append("")
        lines.append("Oblast-level median coefficient diagnostics")
        for predictor in cfg.predictor_cols:
            column = f"coef_map_{predictor}"
            grouped = (
                mapped.dropna(subset=[column])
                .groupby(cfg.oblast_col, dropna=True)[column]
                .median()
                .sort_values()
            )
            if len(grouped):
                lines.append(
                    f"{predictor}: lowest {grouped.index[0]} {grouped.iloc[0]:.4f}; "
                    f"highest {grouped.index[-1]} {grouped.iloc[-1]:.4f}"
                )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_manifest(
    cfg: Config,
    output_dir: Path,
    data_path: Path,
    partition_metadata: Mapping[str, Any],
    final_metadata: Mapping[str, Any],
    use_gpu: bool,
    runtime_seconds: float,
) -> Dict[str, Any]:
    datasets, models = import_gnnwr()
    model_source = Path(inspect.getfile(models)).resolve()
    project_root = model_source
    for parent in [model_source.parent, *model_source.parents]:
        if (parent / ".git").exists():
            project_root = parent
            break
    commit = run_git_command(project_root, "rev-parse", "HEAD")
    dirty = run_git_command(project_root, "status", "--porcelain")
    try:
        gnnwr_version = importlib_metadata.version("gnnwr")
    except importlib_metadata.PackageNotFoundError:
        package = importlib.import_module("gnnwr")
        gnnwr_version = str(getattr(package, "__version__", "unknown"))

    if cfg.require_gnnwr_version and gnnwr_version != cfg.expected_gnnwr_version:
        raise RuntimeError(
            f"Installed GNNWR version is {gnnwr_version}; "
            f"the workflow requires {cfg.expected_gnnwr_version}."
        )
    if gnnwr_version != cfg.expected_gnnwr_version:
        warnings.warn(
            f"Installed GNNWR version {gnnwr_version} differs from the documented "
            f"version {cfg.expected_gnnwr_version}. The manifest records the actual version.",
            RuntimeWarning,
        )
    if commit and commit != cfg.recommended_gnnwr_commit:
        warnings.warn(
            f"Local GNNWR source commit {commit} differs from the recommended "
            f"upstream commit {cfg.recommended_gnnwr_commit}. The manifest records "
            "the actual revision.",
            RuntimeWarning,
        )

    boundary_path = Path(cfg.boundary_path).expanduser().resolve() if cfg.boundary_path else None
    occupied_path = Path(cfg.occupied_path).expanduser().resolve() if cfg.occupied_path else None
    matched_path = Path(cfg.matched_ids_path).expanduser().resolve() if cfg.matched_ids_path else None
    stock_table_path = Path(cfg.stock_table_path).expanduser().resolve() if cfg.stock_table_path else None
    return {
        "configuration": asdict(cfg),
        "partition": dict(partition_metadata),
        "final_model": dict(final_metadata),
        "runtime_seconds": float(runtime_seconds),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": importlib.import_module("sklearn").__version__,
            "scipy": importlib.import_module("scipy").__version__,
            "torch": torch.__version__,
            "cuda_used": bool(use_gpu),
            "cuda_device": torch.cuda.get_device_name(0) if use_gpu else None,
            "gnnwr_version": gnnwr_version,
            "gnnwr_models_source": str(model_source),
            "gnnwr_datasets_source": str(Path(inspect.getfile(datasets)).resolve()),
            "gnnwr_git_commit_if_available": commit,
            "gnnwr_git_dirty_if_available": bool(dirty),
            "recommended_upstream_commit": cfg.recommended_gnnwr_commit,
            "gnnwr_algorithm_doi": GNNWR_PAPER_DOI,
            "gnnwr_package_doi": GNNWR_PACKAGE_DOI,
        },
        "input_hashes": {
            "analytical_table": hash_input_dataset(data_path),
            "matched_ids": hash_input_dataset(matched_path),
            "stock_table": hash_input_dataset(stock_table_path),
            "boundary": hash_input_dataset(boundary_path),
            "occupied_area": hash_input_dataset(occupied_path),
            "script": hash_input_dataset(Path(__file__).resolve()),
        },
        "output_directory": str(output_dir),
    }


# =============================================================================
# Self-test without GNNWR training
# =============================================================================


def run_self_test() -> None:
    """Test data handling, spatial folds, coefficient scaling and Fig. 6."""

    rng = np.random.default_rng(7)
    n = 50
    x_coord = np.repeat(np.arange(10), 5).astype(float) * 100000
    y_coord = np.tile(np.arange(5), 10).astype(float) * 100000
    ratio = np.linspace(0.001, 0.08, n)
    cii = rng.normal(size=n)
    raw = pd.DataFrame(
        {
            "ADM3_PCODE": [f"UA{i:07d}" for i in range(n)],
            "Disp_Rate_FINNAL": ratio,
            "proj_X": x_coord,
            "proj_Y": y_coord,
            "POINT_X": x_coord,
            "POINT_Y": y_coord,
            "CII": cii,
            "CII_pos": cii - cii.min(),
            "Index_Agin": rng.normal(19, 2, size=n),
            "ALL_Damage_Density": rng.gamma(2, 0.2, size=n),
            "Extinguished_Ratio": rng.uniform(0, 100, size=n),
        }
    )
    with tempfile.TemporaryDirectory(prefix="section34_selftest_") as temp:
        temp_path = Path(temp)
        cfg = Config(
            data_path=str(temp_path / "data.xlsx"),
            output_dir=str(temp_path / "output"),
            n_spatial_blocks=10,
            n_spatial_folds=5,
            min_outer_fold_size=5,
            final_seeds=(42,),
        )
        model_df, audit = prepare_model_data(raw, cfg)
        assert len(model_df) == n
        assert int(audit.loc[audit["item"] == "model_rows", "value"].iloc[0]) == n
        assert bool(
            audit.loc[
                audit["item"] == "CII_pos_is_additive_translation", "value"
            ].iloc[0]
        )

        blocks, folds, metadata = create_spatial_blocks_and_folds(model_df, cfg)
        assert len(blocks) == n and len(folds) == n
        assert metadata["actual_spatial_blocks"] == 10

        prep = fit_fold_preprocessor(model_df, cfg.predictor_cols, cfg.target_col)
        coefficient_std = rng.normal(size=(n, len(cfg.predictor_cols) + 1))
        coefficient_raw = back_transform_local_coefficients(coefficient_std, prep)
        x_std = prep.transform_x(model_df, cfg.predictor_cols)
        x_raw = prep.impute_x(model_df, cfg.predictor_cols)
        pred_std = np.sum(coefficient_std[:, :-1] * x_std, axis=1) + coefficient_std[:, -1]
        pred_raw = np.sum(coefficient_raw[:, :-1] * x_raw, axis=1) + coefficient_raw[:, -1]
        assert np.max(np.abs(prep.inverse_y(pred_std) - pred_raw)) < 1e-8

        y = model_df[cfg.target_col].to_numpy()
        x = prep.transform_x(model_df, cfg.predictor_cols)
        beta = fit_ols(x, prep.transform_y(y))
        assert predict_ols(beta, x).shape == (n,)
        gwr = gwr_predict(
            x,
            prep.transform_y(y),
            model_df[["proj_X", "proj_Y"]].to_numpy(),
            x,
            model_df[["proj_X", "proj_Y"]].to_numpy(),
            15,
            1e-8,
        )
        assert gwr.shape == (n,)
        weights = symmetric_knn_weights(
            model_df[["proj_X", "proj_Y"]].to_numpy(), 4
        )
        assert weights.shape == (n, n)

        try:
            import geopandas as gpd
            from shapely.geometry import box
        except ImportError:
            print("SELF-TEST PASSED except optional Fig. 6 rendering because geopandas is absent.")
            return

        boundary = gpd.GeoDataFrame(
            {
                "ADM3_PCODE": raw["ADM3_PCODE"],
                "geometry": [
                    box(x - 40000, y - 40000, x + 40000, y + 40000)
                    for x, y in zip(x_coord, y_coord)
                ],
            },
            crs="EPSG:3035",
        )
        boundary_path = temp_path / "UKR_ADM3_2023.gpkg"
        boundary.to_file(boundary_path, layer="ADM3", driver="GPKG")
        cfg.boundary_path = str(boundary_path)
        cfg.boundary_layer = "ADM3"
        cfg.boundary_id_col = "ADM3_PCODE"
        cfg.figure_dpi = 100
        cfg.output_dir = str(temp_path / "output")
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        coefficient_frame = model_df.copy()
        for index, predictor in enumerate(cfg.predictor_cols):
            coefficient_frame[f"coef_map_{predictor}"] = coefficient_std[:, index]
        figure_paths, _ = create_figure6(
            coefficient_frame, cfg, Path(cfg.output_dir)
        )
        assert all(path.exists() and path.stat().st_size > 0 for path in figure_paths)
    print("SELF-TEST PASSED")


# =============================================================================
# Command line
# =============================================================================


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Section 3.4, Table 3 and Fig. 6 with the official "
            "GNNWR implementation, GWR and OLS."
        )
    )
    parser.add_argument("--data", default="data/Section3_4_GNNWR_input.xlsx")
    parser.add_argument("--output", default="demo_result/section_3_4_gnnwr")
    parser.add_argument("--external-id-col", default="ADM3_PCODE")
    parser.add_argument("--ratio-col", default="Disp_Rate_FINNAL")
    parser.add_argument(
        "--ratio-scale", choices=("proportion", "percent"), default="proportion"
    )
    parser.add_argument("--match-col", default=None)
    parser.add_argument("--matched-ids", default=None)
    parser.add_argument("--matched-id-col", default=None)
    parser.add_argument("--input-is-matched-sample", action="store_true")
    parser.add_argument(
        "--predictors",
        nargs=4,
        default=[
            "CII_pos",
            "Index_Agin",
            "ALL_Damage_Density",
            "Extinguished_Ratio",
        ],
        metavar=("CII", "OLDER", "BD", "NTL_EXT"),
    )
    parser.add_argument("--coords", nargs=2, default=["proj_X", "proj_Y"])

    # Optional raw stock data for the count map in Fig. 6a.
    parser.add_argument("--stock-col", default=None)
    parser.add_argument("--stock-table", default=None)
    parser.add_argument("--stock-table-id-col", default=None)
    parser.add_argument("--population-col", default=None)
    parser.add_argument("--expected-stock-total", type=float, default=None)

    parser.add_argument("--n-spatial-blocks", type=int, default=25)
    parser.add_argument("--n-spatial-folds", type=int, default=5)
    parser.add_argument("--min-outer-fold-size", type=int, default=15)
    parser.add_argument("--dense-layers", type=parse_int_tuple, default=(64, 32))
    parser.add_argument("--final-seeds", type=parse_int_tuple, default=(42,))
    parser.add_argument(
        "--map-coefficient-estimator", choices=("primary", "median"), default="primary"
    )
    parser.add_argument("--moran-k", type=int, default=4)
    parser.add_argument("--moran-permutations", type=int, default=999)

    parser.add_argument("--boundary", default="data/UKR_ADM3_2023.shp")
    parser.add_argument("--boundary-layer", default=None)
    parser.add_argument("--boundary-id-col", default="ADM3_PCODE")
    parser.add_argument("--oblast-col", default=None)
    parser.add_argument("--occupied", default=None)
    parser.add_argument("--occupied-layer", default=None)
    parser.add_argument("--figure-dpi", type=int, default=600)
    parser.add_argument("--skip-figure", action="store_true")

    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-other-gnnwr-version",
        action="store_true",
        help="Allow execution with a GNNWR version other than 0.1.17; the actual version is recorded in the manifest.",
    )
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def resolve_path(text: Optional[str]) -> Optional[Path]:
    if text is None:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    data_path = resolve_path(args.data)
    output_dir = resolve_path(args.output)
    assert data_path is not None and output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        data_path=str(data_path),
        output_dir=str(output_dir),
        external_id_col=args.external_id_col,
        ratio_col=args.ratio_col,
        stock_col=args.stock_col,
        stock_table_path=str(resolve_path(args.stock_table)) if args.stock_table else None,
        stock_table_id_col=args.stock_table_id_col,
        population_col=args.population_col,
        match_col=args.match_col,
        matched_ids_path=str(resolve_path(args.matched_ids)) if args.matched_ids else None,
        matched_id_col=args.matched_id_col,
        input_is_matched_sample=args.input_is_matched_sample,
        ratio_scale=args.ratio_scale,
        spatial_cols=tuple(args.coords),
        predictor_cols=tuple(args.predictors),
        expected_stock_total=args.expected_stock_total,
        n_spatial_blocks=args.n_spatial_blocks,
        n_spatial_folds=args.n_spatial_folds,
        min_outer_fold_size=args.min_outer_fold_size,
        gnnwr_dense_layers=tuple(args.dense_layers),
        final_seeds=tuple(args.final_seeds),
        map_coefficient_estimator=args.map_coefficient_estimator,
        moran_k=args.moran_k,
        moran_permutations=args.moran_permutations,
        boundary_path=str(resolve_path(args.boundary)) if args.boundary else None,
        boundary_layer=args.boundary_layer,
        boundary_id_col=args.boundary_id_col,
        oblast_col=args.oblast_col,
        occupied_path=str(resolve_path(args.occupied)) if args.occupied else None,
        occupied_layer=args.occupied_layer,
        figure_dpi=args.figure_dpi,
        skip_figure=args.skip_figure,
        require_gnnwr_version=not args.allow_other_gnnwr_version,
    )
    if cfg.n_spatial_folds < 2:
        raise ValueError("At least two spatial folds are required.")
    if cfg.n_spatial_blocks < cfg.n_spatial_folds:
        raise ValueError("The number of spatial blocks must be at least the number of folds.")
    if len(set(cfg.final_seeds)) != len(cfg.final_seeds):
        raise ValueError("--final-seeds contains duplicate values.")

    if args.quick_test:
        cfg.gnnwr_max_epoch = 120
        cfg.gnnwr_early_stop = 20
        cfg.moran_permutations = 99
        cfg.figure_dpi = min(cfg.figure_dpi, 150)
        print("QUICK-TEST MODE IS ACTIVE. Do not report these metrics in the manuscript.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    use_gpu = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )

    print(f"Repository root : {REPO_ROOT}")
    print(f"Script directory : {SCRIPT_DIR}")
    print(f"Input table      : {data_path}")
    print(f"Output directory : {output_dir}")
    print(f"CUDA used        : {use_gpu}")

    raw = safe_read_table(data_path)
    model_df, audit = prepare_model_data(raw, cfg)
    correlation, vif = predictor_diagnostics(model_df, cfg)
    audit.to_csv(output_dir / "Data_audit.csv", index=False, encoding="utf-8-sig")
    correlation.to_csv(output_dir / "Predictor_correlation.csv", encoding="utf-8-sig")
    vif.to_csv(output_dir / "Predictor_VIF.csv", index=False, encoding="utf-8-sig")
    model_df.to_csv(
        output_dir / "Analytical_sample_used_for_GNNWR.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\nDATA AUDIT")
    print(audit.to_string(index=False))

    if args.audit_only:
        print("\nAudit completed. No model was fitted.")
        return

    datasets, models = import_gnnwr()
    print(f"GNNWR models source   : {inspect.getfile(models)}")
    print(f"GNNWR datasets source : {inspect.getfile(datasets)}")

    started = time.time()
    set_global_seed(cfg.random_seed)
    predictions, fold_metrics, fold_assignments, gwr_selections, partition_metadata = run_spatial_cross_validation(
        model_df,
        cfg,
        use_gpu,
        output_dir,
    )
    table3_detailed, table3_rounded, ratio_metrics = build_table3(
        predictions,
        cfg,
        partition_metadata["actual_spatial_blocks"],
    )

    coefficients, all_seed_coefficients, stability, final_metadata = fit_final_gnnwr_and_extract_coefficients(
        model_df,
        fold_metrics,
        cfg,
        use_gpu,
        output_dir,
    )
    coefficients = attach_optional_stock_table(coefficients, cfg)
    coefficient_stats = coefficient_summary(coefficients, cfg)

    table3_detailed.to_csv(output_dir / "Table3_model_performance.csv", index=False, encoding="utf-8-sig")
    table3_rounded.to_csv(output_dir / "Table3_model_performance_rounded.csv", index=False, encoding="utf-8-sig")
    ratio_metrics.to_csv(output_dir / "Model_performance_ratio_scale_supplement.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_dir / "All_models_spatial_OOF_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(output_dir / "Model_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    fold_assignments.to_csv(output_dir / "Spatial_fold_assignments.csv", index=False, encoding="utf-8-sig")
    gwr_selections.to_csv(output_dir / "GWR_inner_validation_bandwidth_search.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(output_dir / "GNNWR_local_coefficients_primary.csv", index=False, encoding="utf-8-sig")
    all_seed_coefficients.to_csv(output_dir / "GNNWR_local_coefficients_all_seeds.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(output_dir / "GNNWR_local_coefficient_seed_stability.csv", index=False, encoding="utf-8-sig")
    coefficient_stats.to_csv(output_dir / "GNNWR_local_coefficient_summary.csv", index=False, encoding="utf-8-sig")

    figure_paths: List[Path] = []
    mapped = None
    if cfg.skip_figure:
        warnings.warn(
            "Fig. 6 generation was explicitly skipped. This run does not reproduce "
            "all Section 3.4 outputs.",
            RuntimeWarning,
        )
    else:
        figure_paths, mapped = create_figure6(coefficients, cfg, output_dir)
        mapped.drop(columns="geometry").to_csv(
            output_dir / "Figure6_joined_attributes.csv",
            index=False,
            encoding="utf-8-sig",
        )

    write_section34_summary(
        model_df,
        table3_detailed,
        coefficient_stats,
        final_metadata,
        cfg,
        output_dir / "Section3_4_results_summary.txt",
        mapped,
    )

    runtime = time.time() - started
    manifest = collect_manifest(
        cfg,
        output_dir,
        data_path,
        partition_metadata,
        final_metadata,
        use_gpu,
        runtime,
    )
    with (output_dir / "Run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 88)
    print("TABLE 3")
    print("=" * 88)
    print(table3_rounded.to_string(index=False))
    print("\nOUTPUT DIRECTORY")
    print(output_dir)
    for path in figure_paths:
        print(path)
    print(f"\nCompleted in {runtime / 60.0:.2f} minutes.")


if __name__ == "__main__":
    main()
