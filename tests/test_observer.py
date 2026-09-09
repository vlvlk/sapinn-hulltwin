"""Identifiability test: can the PINN observer recover k_s(t) and k_p(t)
from noisy daily logs of the synthetic vessel?"""

import numpy as np
import pandas as pd
import pytest
import torch

from hulltwin.data.synthetic import simulate_voyage
from hulltwin.pinn.observer import HullTwinObserver

N_DAYS = 360


@pytest.fixture(scope="module")
def trained() -> tuple[HullTwinObserver, pd.DataFrame]:
    torch.manual_seed(0)
    np.random.seed(0)
    df = simulate_voyage(n_days=N_DAYS, cleanings=(180,), seed=3)
    observer = HullTwinObserver(n_days=N_DAYS)
    observer.fit(df, epochs=4000, lr=5e-2)
    return observer, df


def _batch(df: pd.DataFrame, stride: int = 3) -> dict[str, torch.Tensor]:
    idx = np.arange(0, len(df), stride)
    return {
        "v": torch.tensor(df["v_meas"].to_numpy()[idx], dtype=torch.float32),
        "n": torch.tensor(df["rpm_meas"].to_numpy()[idx] / 60.0, dtype=torch.float32),
        "hs": torch.tensor(df["hs"].to_numpy()[idx], dtype=torch.float32),
        "mu_rad": torch.tensor(
            np.radians(df["wave_heading_deg"].to_numpy()[idx]), dtype=torch.float32
        ),
        "torque": torch.tensor(df["torque_meas"].to_numpy()[idx], dtype=torch.float32),
        "day": torch.tensor(idx, dtype=torch.long),
        "cleaning": torch.tensor(df["cleaning_today"].to_numpy()[idx] > 0.5),
    }


def test_ks_mean_trajectory_recovered(trained: tuple[HullTwinObserver, pd.DataFrame]):
    observer, df = trained
    days = np.arange(N_DAYS)
    pred = observer.predict_ks_mean(days)
    true = df["ks_mean_true"].to_numpy() * 1e6  # -> um
    corr = np.corrcoef(pred, true)[0, 1]
    assert corr > 0.85, f"ks trajectory correlation too low: {corr:.3f}"
    # end-to-end error of the fitted trajectory
    assert np.median(np.abs(pred - true) / true) < 0.15


def test_cleaning_detected_as_field_reset(trained: tuple[HullTwinObserver, pd.DataFrame]):
    observer, df = trained
    pred = observer.predict_ks_mean(np.arange(N_DAYS))
    before = pred[170:179].mean()  # aged hull just before cleaning (day 180)
    after = pred[182:191].mean()  # fresh hull just after
    assert after < 0.7 * before, "observer did not detect the hull cleaning"


def test_propeller_degradation_recovered(trained: tuple[HullTwinObserver, pd.DataFrame]):
    observer, df = trained
    days = np.arange(N_DAYS)
    pred_kp = observer.predict_kp(days)
    true_kp = df["kp_true"].to_numpy()
    rel_err = abs(pred_kp[-1] - true_kp[-1]) / true_kp[-1]
    assert rel_err < 0.10, f"final kp relative error too high: {rel_err:.1%}"
    # monotone degradation
    assert np.all(np.diff(pred_kp) >= -1e-6)


def test_physics_residuals_small_after_training(trained: tuple[HullTwinObserver, pd.DataFrame]):
    observer, df = trained
    with torch.no_grad():
        res = observer.physics_residuals(_batch(df))
    # The thrust residual retains the noise floor (~1.5-2% 1-sigma from speed
    # and rpm noise): the smoothness term deliberately refuses to chase
    # per-day noise, so mean |r| ~ 3-4% is the healthy operating point.
    assert res["thrust"].abs().mean() < 0.04
    assert res["torque"].abs().mean() < 0.03


def test_field_shape_is_prior_based(trained: tuple[HullTwinObserver, pd.DataFrame]):
    """The spatial field is a prior-based diagnostic: stern rougher than bow
    (stern_bias_prior), scaling with the identified per-day mean."""
    observer, _ = trained
    field = observer.predict_ks_field(np.array([100, 300]), n_x=101)  # (101, 2)
    assert field.shape == (101, 2)
    assert field[:, 1].mean() > field[:, 0].mean()  # fouling grew between days
    assert field[-20:, 0].mean() > field[:20, 0].mean()  # stern > bow
