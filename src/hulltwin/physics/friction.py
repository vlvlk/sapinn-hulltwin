"""Friction correlations: ITTC-1957 line and Bowden-Davison roughness allowance.

References
----------
- ITTC 1957 model-ship correlation line: C_F0 = 0.075 / (log10(Re) - 2)^2
- Bowden & Davison (1974) roughness allowance:
  Delta CF = 0.105 * (k_s / L)^(1/3) - 0.64e-3, clamped at >= 0.
"""

from __future__ import annotations

import numpy as np

#: Kinematic viscosity of seawater at ~15 C [m^2/s]
NU_SEAWATER = 1.05e-6


def cf_ittc57(re: float | np.ndarray) -> float | np.ndarray:
    """ITTC-1957 smooth-hull friction coefficient.

    Parameters
    ----------
    re : Reynolds number, Re = U * x / nu (must be > 10**2.1 for the log to stay positive).

    Returns
    -------
    C_F0 = 0.075 / (log10(Re) - 2)^2
    """
    re = np.asarray(re, dtype=float)
    if np.any(re <= 10.0**2.1):
        raise ValueError("Reynolds number below validity range of ITTC-1957 line")
    log_re = np.log10(re)
    return 0.075 / (log_re - 2.0) ** 2  # type: ignore[no-any-return]


def roughness_allowance(ks: float | np.ndarray, ship_length: float) -> float | np.ndarray:
    """Bowden-Davison roughness allowance Delta CF (ship-scale, based on hull length).

    Parameters
    ----------
    ks : equivalent sand-grain roughness [m] (0 = hydraulically smooth)
    ship_length : ship length between perpendiculars [m]

    Returns
    -------
    Delta CF = max(0, 0.105 * (ks/L)^(1/3) - 0.64e-3)
    """
    ks = np.asarray(ks, dtype=float)
    if np.any(ks < 0.0):
        raise ValueError("ks must be non-negative")
    if ship_length <= 0.0:
        raise ValueError("ship length must be positive")
    dcf = 0.105 * (ks / ship_length) ** (1.0 / 3.0) - 0.64e-3
    return np.asarray(np.maximum(dcf, 0.0))


def cf_local(
    x: float | np.ndarray,
    v: float,
    ks: float | np.ndarray = 0.0,
    nu: float = NU_SEAWATER,
) -> float | np.ndarray:
    """Local skin-friction coefficient at streamwise station ``x``.

    Engineering surrogate for the boundary-layer strip model:
    smooth part follows the ITTC-1957 shape evaluated at local Re_x, the
    roughness part scales the Bowden-Davison form with the local length x
    instead of the ship length (Delta cf ~ (ks/x)^(1/3)).

    Parameters
    ----------
    x : streamwise coordinate(s) from the leading edge [m], x > 0
    v : flow speed [m/s], v > 0
    ks : local equivalent sand roughness [m] (may vary with x)
    nu : kinematic viscosity [m^2/s]

    Returns
    -------
    Local C_f(x, Re_x, ks) (1D, same shape as x / ks broadcast).
    """
    x = np.asarray(x, dtype=float)
    ks = np.asarray(ks, dtype=float)
    if np.any(x <= 0.0):
        raise ValueError("x must be positive (distance from leading edge)")
    if v <= 0.0:
        raise ValueError("speed must be positive")

    re_x = v * x / nu
    cf0 = np.asarray(cf_ittc57(re_x))
    dcf = np.asarray(0.105 * (ks / x) ** (1.0 / 3.0) - 0.64e-3)
    dcf = np.maximum(dcf, 0.0)
    return np.asarray(cf0 + dcf)
