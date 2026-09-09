"""Propeller open-water model (quadratic K_T / K_Q polynomials) and torque residual.

Defaults approximate a Wageningen B4-70 propeller with P/D = 1.0 and must be
replaced by the vessel-specific polynomial (Oosterveld & van Oossanen
regression) in the calibration stage.

Propeller degradation (blade fouling / erosion) enters as a multiplicative
torque penalty ``kp`` >= 1: a fouled blade produces the same thrust with more
torque, KQ_effective = KQ * kp. This gives the independent identification
equation separating propeller wear from hull fouling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PropellerParams:
    """Quadratic open-water polynomial coefficients for K_T and K_Q.

    Defaults are a least-squares fit to representative B4-70 (P/D = 1.0)
    open-water points: K_T(0)=0.52, K_T(0.5)=0.30, K_T(0.9)=0.06;
    K_Q(0)=0.053, K_Q(0.5)=0.031, K_Q(0.9)=0.013.
    """

    kt_coeffs: tuple[float, float, float] = (0.520, -0.3511, -0.1778)  # a0 + a1 J + a2 J^2
    kq_coeffs: tuple[float, float, float] = (0.0530, -0.04344, -0.00111)  # b0 + b1 J + b2 J^2
    diameter: float = 6.0  # [m]
    rho: float = 1025.0  # water density [kg/m^3]


def _poly(j: np.ndarray, coeffs: tuple[float, float, float]) -> np.ndarray:
    a0, a1, a2 = coeffs
    return a0 + a1 * j + a2 * j * j


def kt(j: float | np.ndarray, params: PropellerParams | None = None) -> np.ndarray:
    """Thrust coefficient K_T(J). Blade degradation is intentionally NOT put
    here: all degradation is carried by the K_Q torque penalty ``kp`` so that
    the torque identity stays a clean independent equation."""
    params = params or PropellerParams()
    j = np.asarray(j, dtype=float)
    return _poly(j, params.kt_coeffs)


def kq(
    j: float | np.ndarray,
    params: PropellerParams | None = None,
    kp: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Torque coefficient K_Q(J) with degradation penalty ``kp`` >= 1.

    KQ_effective = KQ_clean * kp
    """
    params = params or PropellerParams()
    j = np.asarray(j, dtype=float)
    kp = np.asarray(kp, dtype=float)
    if np.any(kp < 1.0):
        raise ValueError("kp (degradation penalty) must be >= 1")
    return _poly(j, params.kq_coeffs) * kp


def open_water_efficiency(
    j: float | np.ndarray,
    params: PropellerParams | None = None,
    kp: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Open-water efficiency eta_0 = J * K_T / (2 pi K_Q)."""
    j = np.asarray(j, dtype=float)
    if np.any(j <= 0.0):
        raise ValueError("advance ratio J must be positive")
    kt_v = kt(j, params)
    kq_v = kq(j, params, kp)
    return j * kt_v / (2.0 * np.pi * kq_v)


def torque_residual(
    q_measured: float | np.ndarray,
    n: float | np.ndarray,
    j: float | np.ndarray,
    params: PropellerParams | None = None,
    kp: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Torque identity residual: Q_measured - rho * n^2 * D^5 * K_Q(J) * kp.

    Independent of the hull module — the equation that makes propeller
    degradation identifiable separately from hull fouling.
    Units: n [rev/s], Q [N*m].
    """
    params = params or PropellerParams()
    n = np.asarray(n, dtype=float)
    j = np.asarray(j, dtype=float)
    if np.any(n <= 0.0):
        raise ValueError("shaft revolution rate n must be positive (rev/s)")
    q_hat = params.rho * n**2 * params.diameter**5 * kq(j, params, kp)
    return np.asarray(q_measured, dtype=float) - q_hat
