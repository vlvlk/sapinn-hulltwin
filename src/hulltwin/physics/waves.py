"""Added resistance in waves (calibratable Kreitner-type heuristic).

    R_aw = c_aw * rho * g * (H_{1/3} / 2)^2 * B^2 / L * f(mu)

where f(mu) is a heading factor (1.0 head seas, decreasing for following
seas). ``c_aw`` is treated as a calibratable parameter — the PINN observer
absorbs its uncertainty together with the sea-state estimate from IMU data.
"""

from __future__ import annotations

import numpy as np


def heading_factor(mu_deg: float | np.ndarray) -> np.ndarray:
    """Heading factor f(mu): 1.0 for head seas (mu = 180 deg relative),
    0.4 for beam, 0.2 for following seas. Smooth cosine blend."""
    mu = np.radians(np.asarray(mu_deg, dtype=float))
    # mu measured from head seas (180 deg = head-on in nautical convention is
    # confusing; here mu = 0 means head seas by definition of the caller).
    return np.clip(0.6 + 0.4 * np.cos(mu), 0.2, 1.0)  # type: ignore[no-any-return]


def added_resistance_aw(
    hs: float | np.ndarray,
    beam: float,
    ship_length: float,
    rho: float = 1025.0,
    g: float = 9.81,
    c_aw: float = 0.5,
    mu_deg: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Added wave resistance [N].

    Parameters
    ----------
    hs : significant wave height [m], >= 0
    beam : ship beam [m]
    ship_length : ship length [m]
    rho : water density [kg/m^3]
    g : gravity [m/s^2]
    c_aw : calibratable coefficient (order 0.5 for typical merchant hulls)
    mu_deg : wave heading relative to the vessel (0 = head seas)
    """
    hs = np.asarray(hs, dtype=float)
    if np.any(hs < 0.0):
        raise ValueError("significant wave height must be non-negative")
    if beam <= 0.0 or ship_length <= 0.0:
        raise ValueError("beam and length must be positive")
    raw = c_aw * rho * g * (hs / 2.0) ** 2 * beam**2 / ship_length
    return raw * heading_factor(mu_deg)  # type: ignore[no-any-return]
