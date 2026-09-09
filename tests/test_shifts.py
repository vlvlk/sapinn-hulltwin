"""Tests for the Shifts power-consumption adapter (roadmap stage 5)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hulltwin.data import shifts
from hulltwin.data.shifts import (
    ShiftsPowerModel,
    load_split,
    median_relative_error,
    run_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/external/power_consumption_upload/synthetic_data"


def _tiny_shifts_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Raw-schema Shifts-like frame (exact CSV column names)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "draft_aft_telegram": rng.uniform(6.0, 9.0, n),
            "draft_fore_telegram": rng.uniform(6.0, 9.0, n),
            "stw": rng.uniform(9.0, 14.0, n),
            "diff_speed_overground": rng.uniform(-0.5, 1.5, n),
            "awind_vcomp_provider": rng.normal(0, 4, n),
            "awind_ucomp_provider": rng.normal(0, 4, n),
            "rcurrent_vcomp": rng.normal(0, 0.3, n),
            "rcurrent_ucomp": rng.normal(0, 0.3, n),
            "comb_wind_swell_wave_height": rng.uniform(0.1, 4.0, n),
            "timeSinceDryDock": rng.uniform(0, 3000, n),
            "time_id": np.arange(n, dtype=float),
            "power": rng.uniform(4000, 9000, n),
        }
    )


def test_load_split_synthetic_train_has_adapter_schema() -> None:
    df = load_split("train", DATA_DIR)
    expected = {
        "draft_mean",
        "v_ms",
        "wind_speed",
        "current_speed",
        "age",
        "ks_m",
        "power",
        "hs",
        "stw",
    }
    assert expected <= set(df.columns)
    assert (df["v_ms"] > 0).all()
    assert (df["ks_m"] >= 30e-6).all()  # base roughness right after drydock
    assert len(df) == 523190  # full released synthetic train split


def test_load_split_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_split("train", tmp_path)


def test_design_matrix_is_nonnegative_and_finite() -> None:
    df = load_split("train", DATA_DIR).head(500)
    a = ShiftsPowerModel().design_matrix(df)
    assert np.isfinite(a).all()
    assert (a >= 0).all()  # V^3, friction power, wave power, draft: all >= 0


def test_model_recovers_power_generated_by_same_structure() -> None:
    """If power is *generated* by the same V^3 + wave + draft structure
    (cf_scale = 0), the calibrated model must fit it almost exactly —
    a sanity check of the projected least-squares fitter."""
    raw = _tiny_shifts_frame(300, seed=1)
    v = raw["stw"].to_numpy() * 0.514444
    hs = raw["comb_wind_swell_wave_height"].to_numpy()
    draft_mean = 0.5 * (
        raw["draft_aft_telegram"].to_numpy() + raw["draft_fore_telegram"].to_numpy()
    )
    p_wave = 1025 * 9.81**2 * hs**2 * 900 / (32 * 180) * v / 1000
    raw["power"] = 40.0 * v**3 + p_wave + 30.0 * draft_mean

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic_data" / "train.csv"
        path.parent.mkdir(parents=True)
        raw.to_csv(path, index=False)
        df = load_split("train", tmp)

    y = df["power"].to_numpy()
    model = ShiftsPowerModel().fit(df, y)
    assert median_relative_error(y, model.predict(df)) < 0.01
    assert model._coef is not None
    np.testing.assert_allclose(model._coef[0], 40.0, rtol=0.2)  # calm V^3 coeff
    np.testing.assert_allclose(model._coef[2], 1.0, rtol=0.2)  # wave scale
    np.testing.assert_allclose(model._coef[3], 30.0, rtol=0.2)  # draft gain


def test_predict_before_fit_raises() -> None:
    df = load_split("train", DATA_DIR).head(50)
    with pytest.raises(RuntimeError, match="not fitted"):
        ShiftsPowerModel().predict(df)


def test_benchmark_runs_on_release_and_beats_naive_v3() -> None:
    """Stage-5 acceptance: the physics model must beat a pure V^3 baseline
    (which ignores fouling and weather) on the in-domain dev split."""
    metrics = run_benchmark(DATA_DIR, verbose=False)
    assert metrics["dev_in_median_err"] > 0
    assert metrics["dev_out_median_err"] > 0

    train = load_split("train", DATA_DIR)
    dev_in = load_split("dev_in", DATA_DIR)
    v3 = train["v_ms"].to_numpy() ** 3
    k = float(np.dot(v3, train["power"]) / np.dot(v3, v3))
    err_naive = median_relative_error(
        dev_in["power"].to_numpy(), k * dev_in["v_ms"].to_numpy() ** 3
    )
    assert metrics["dev_in_median_err"] < err_naive


def test_median_relative_error_zero_true_does_not_crash() -> None:
    """Zero measured power must not divide-by-zero: it yields a huge (finite)
    relative error which the median across the split absorbs."""
    err = median_relative_error(np.array([0.0, 10.0]), np.array([1.0, 12.0]))
    assert np.isfinite(err)
    # |1/1e-9| = 1e9 for the zero-true sample, 0.2 for the other; median of two
    assert err == pytest.approx(0.5 * (1e9 + 0.2))
    assert median_relative_error(np.array([10.0, 20.0]), np.array([10.0, 30.0])) == pytest.approx(
        0.5 * (0.0 + 0.5)
    )
    # verify the unit-conversion constant used by the adapter
    assert shifts._KN_TO_MS == 0.514444
