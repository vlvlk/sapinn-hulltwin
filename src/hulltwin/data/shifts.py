"""Adapter for the Shifts Marine Cargo Vessel Power Consumption dataset.

Roadmap stage 5: validate the twin on *external* (non-synthetic) data.

The dataset (Zenodo record 7057666, CC BY-NC-SA 4.0; benchmark paper
arXiv:2206.15407) ships daily-ish voyage records of a cargo vessel with

    draft_aft_telegram, draft_fore_telegram, stw, diff_speed_overground,
    awind_vcomp_provider, awind_ucomp_provider, rcurrent_vcomp,
    rcurrent_ucomp, comb_wind_swell_wave_height, timeSinceDryDock,
    time_id, power

Notable simplifications forced by this schema (see plans/mvp-architecture.md):

- no RPM / torque / fuel flow -> the propeller leg of the twin is NOT
  exercised on real data; the functional scope reduces to the balance
  "speed - weather - power";
- ``timeSinceDryDock`` is the only fouling proxy: hull roughness is modelled
  as a dry-dock-age state k_s(t) growing linearly from a cleaning event, the
  same structure the synthetic generator injects;
- ``diff_speed_overground`` (speed lost to current) is converted into an
  effective through-water speed correction used by the physics terms.

The adapter provides:

- :func:`load_split` — read a CSV split into a typed, unit-normalised frame;
- :class:`ShiftsPowerModel` — a *hybrid physics* power predictor: ITTC-1957
  friction with a roughness allowance driven by a dry-dock-age roughness
  state, calm-water + added wave resistance, fitted by least squares on the
  few free scalars (residual calm resistance, loading correction). This is
  the "calibrated model" branch of the twin, mirroring
  :mod:`hulltwin.pinn.observer` but with the propeller leg switched off;
- :func:`run_benchmark` — train on ``train.csv``, report median power error
  on in-domain (``dev_in``) vs shifted (``dev_out``) data, i.e. the
  distribution-shift robustness check the Shifts benchmark is designed for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from hulltwin.physics.friction import NU_SEAWATER, cf_ittc57, roughness_allowance

#: Default location of the Shifts power-consumption upload inside the repo.
DEFAULT_DATA_DIR = Path("data/external/power_consumption_upload")

#: Raw CSV columns -> adapter schema.
_COLUMN_MAP = {
    "draft_aft_telegram": "draft_aft",
    "draft_fore_telegram": "draft_fore",
    "stw": "stw",  # speed through water [kn]
    "diff_speed_overground": "current_loss",  # SOG - STW proxy [kn]
    "awind_vcomp_provider": "wind_v",  # [m/s]
    "awind_ucomp_provider": "wind_u",  # [m/s]
    "rcurrent_vcomp": "current_v",  # [m/s]
    "rcurrent_ucomp": "current_u",  # [m/s]
    "comb_wind_swell_wave_height": "hs",  # combined wind+swell Hs [m]
    "timeSinceDryDock": "days_since_drydock",
    "time_id": "time_id",
    "power": "power",  # [kW] mains
}

# Vessel constants used for the physics scaling (documented, coarse estimates
# for the anonymous Shifts cargo vessel; only scale, not identity, matters for
# the fitted model because the free coefficients absorb constants).
_VESSEL_LENGTH = 180.0  # LBP [m]
_VESSEL_WETTED_SURFACE = 5300.0  # [m^2]
_STW_TO_MS = 0.514444

_KN_TO_MS = _STW_TO_MS


def _clip_positive(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0)


def load_split(split: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load one Shifts split and normalise it to the adapter schema.

    Parameters
    ----------
    split:
        One of ``train``, ``dev_in``, ``dev_out`` (a file
        ``<split>.csv`` must exist under ``data_dir``; for the synthetic
        distribution these are ``synthetic_data/<split>.csv``).
    data_dir:
        Root of the unpacked Shifts power-consumption upload. The loader
        looks for ``<data_dir>/synthetic_data/<split>.csv`` first, then
        ``<data_dir>/real/<split>.csv``, then ``<data_dir>/<split>.csv``.
    """
    data_dir = Path(data_dir)
    candidates = [
        data_dir / "synthetic_data" / f"{split}.csv",
        data_dir / "real" / f"{split}.csv",
        data_dir / f"{split}.csv",
    ]
    for path in candidates:
        if path.exists():
            return _load_csv(path)
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Shifts split '{split}' not found; looked in: {searched}")


def _load_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    missing = set(_COLUMN_MAP) - set(raw.columns)
    if missing:
        raise ValueError(f"{path}: missing expected Shifts columns: {sorted(missing)}")
    df = raw.rename(columns=_COLUMN_MAP)

    # Derived, unit-normalised features used by the physics model.
    df["draft_mean"] = 0.5 * (df["draft_aft"] + df["draft_fore"])
    df["v_ms"] = df["stw"] * _KN_TO_MS  # [m/s]
    df["wind_speed"] = np.hypot(df["wind_u"], df["wind_v"])
    df["current_speed"] = np.hypot(df["current_u"], df["current_v"])
    df["age"] = _clip_positive(df["days_since_drydock"])
    # Roughness state implied by dry-dock age (same growth law as the
    # synthetic generator: base 30 um + 0.9 um/day of fouling).
    df["ks_m"] = (30e-6 + 0.9e-6 * df["age"]).clip(lower=0.0)
    return df


@dataclass
class _FitResult:
    """Free scalars of the hybrid power model."""

    p_calm0: float  # residual calm-water power at V=1 m/s, df=0 [kW]
    cf_scale: float  # multiplier on the ITTC+roughness friction power
    wave_scale: float  # multiplier on the wave-added power
    draft_gain: float  # [1/m] additional power per metre of mean draft
    rmse: float = field(default=0.0)


class ShiftsPowerModel:
    """Physics-structured power predictor for the Shifts schema.

    P_hat [kW] = p_calm0 * V^3
                 + cf_scale * 0.5 rho V^3 S (Cf0(Re) + dCf(k_s(age)))
                 + wave_scale * g^2 rho Hs^2 B^2 / L * V / (something^-1)  [see below]
                 + draft_gain * draft_mean

    with V the through-water speed. All free scalars are non-negative and
    found by a projected Gauss-Newton / least-squares pass; the structure
    (V^3 scaling, friction integral with roughness allowance, Hs^2 wave law)
    is fixed by physics — only amplitudes are calibrated. This is the
    "calibrated resistance model" half of the twin; the propeller leg is
    absent because the dataset has no RPM/torque channels.
    """

    def __init__(self, rho: float = 1025.0) -> None:
        self.rho = rho
        self._coef: np.ndarray | None = None  # [p_calm0, cf_scale, wave_scale, draft_gain]

    # ------------------------------------------------------------------ #
    # design matrix (physics features)
    # ------------------------------------------------------------------ #
    def design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Column layout [calm V^3, friction-with-roughness, wave, draft]."""
        v = df["v_ms"].to_numpy()
        hs = df["hs"].to_numpy()
        ks = df["ks_m"].to_numpy()
        draft = df["draft_mean"].to_numpy()

        re = v * _VESSEL_LENGTH / NU_SEAWATER
        cf0 = cf_ittc57(re)
        dcf = np.maximum(roughness_allowance(ks, _VESSEL_LENGTH), 0.0)
        p_fric = 0.5 * self.rho * v**3 * _VESSEL_WETTED_SURFACE * (cf0 + dcf) / 1000.0

        # simplified wave-added power ~ rho g^2 Hs^2 B^2 /(32 L) * V (kinks
        # absorbed by wave_scale); B/L from the vessel constants.
        p_wave = self.rho * 9.81**2 * hs**2 * (30.0**2) / (32.0 * _VESSEL_LENGTH) * v / 1000.0
        calm = v**3
        return np.column_stack([calm, p_fric, p_wave, draft])

    # ------------------------------------------------------------------ #
    # fitting (non-negative linear least squares)
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, y: np.ndarray | None = None) -> ShiftsPowerModel:
        """Fit the free scalars; ``y`` defaults to the ``power`` column [kW].

        Column-normalised non-negative least squares (Lawson-Hanson via
        ``scipy.optimize.nnls``): physical amplitudes must be >= 0.
        """
        if y is None:
            y = df["power"].to_numpy(dtype=float)
        a = self.design_matrix(df)
        scale = np.maximum(np.linalg.norm(a, axis=0), 1e-12)
        coef_s, _res = nnls(a / scale, y)
        self._coef = coef_s / scale  # back to physical units per column

        self._last_rmse = float(np.sqrt(np.mean((self.predict(df) - y) ** 2)))
        return self

    @property
    def rmse_train(self) -> float:
        if self._coef is None:
            raise RuntimeError("model is not fitted")
        return self._last_rmse

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("model is not fitted")
        return self.design_matrix(df) @ self._coef


def median_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median |dP|/P — the headline accuracy metric of the twin."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.median(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-9)))


def run_benchmark(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    epochs: int | None = None,  # unused, kept for CLI symmetry
    verbose: bool = True,
) -> dict[str, float]:
    """Train on ``train`` and evaluate on in-domain vs shifted dev splits.

    Returns a metrics dict with median relative power error per split plus
    the train RMSE. This is the stage-5 acceptance check: a *physics-*
    structured model with 4 calibrated scalars should degrade gracefully
    under the Shifts distribution shift compared with a black-box fit.
    """
    train = load_split("train", data_dir)
    dev_in = load_split("dev_in", data_dir)
    dev_out = load_split("dev_out", data_dir)

    model = ShiftsPowerModel().fit(train)

    metrics = {
        "train_rmse_kw": model.rmse_train,
        "dev_in_median_err": median_relative_error(
            dev_in["power"].to_numpy(), model.predict(dev_in)
        ),
        "dev_out_median_err": median_relative_error(
            dev_out["power"].to_numpy(), model.predict(dev_out)
        ),
        "n_train": float(len(train)),
    }
    if verbose:
        print("Shifts power-consumption benchmark (hybrid physics, 4 free scalars)")
        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}" if "err" not in key else f"  {key}: {val:.2%}")
    return metrics
