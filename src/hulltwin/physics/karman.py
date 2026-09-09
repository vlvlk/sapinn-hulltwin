"""Integral von Karman momentum equation along a streamwise strip.

    dtheta/dx + (2 + H) * (theta / Ue) * dUe/dx = Cf(x) / 2

For the MVP the shape parameter H is held constant (fully turbulent flow,
H ~ 1.3-1.5); ``Ue(x)`` may be an arbitrary external-velocity profile along
the strip (e.g. from a hull-line potential solution or a simple potential
approximation around the hull).

This module provides:
- ``karman_residual``: PDE residual used both by the synthetic generator and
  (later) by the PINN loss term L_BL;
- ``solve_momentum_boundary_layer``: classical forward solve (scipy IVP) used
  to generate ground-truth theta(x) for synthetic data.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp


def karman_residual(
    theta: float | np.ndarray,
    dtheta_dx: float | np.ndarray,
    ue: float | np.ndarray,
    due_dx: float | np.ndarray,
    cf: float | np.ndarray,
    h: float = 1.4,
) -> np.ndarray:
    """Residual of the momentum integral equation (should be ~0 for a solution).

    Parameters
    ----------
    theta : momentum thickness [m]
    dtheta_dx : d(theta)/dx [m/m]
    ue : external-velocity [m/s], > 0
    due_dx : d(Ue)/dx [1/s]
    cf : local skin-friction coefficient at the same x
    h : shape parameter (constant, turbulent)
    """
    theta = np.asarray(theta, dtype=float)
    dtheta_dx = np.asarray(dtheta_dx, dtype=float)
    ue = np.asarray(ue, dtype=float)
    due_dx = np.asarray(due_dx, dtype=float)
    cf = np.asarray(cf, dtype=float)
    if np.any(ue <= 0.0):
        raise ValueError("ue must be positive")
    return dtheta_dx + (2.0 + h) * (theta / ue) * due_dx - cf / 2.0


def solve_momentum_boundary_layer(
    x_grid: np.ndarray,
    ue_func: Callable[[np.ndarray], np.ndarray],
    cf_func: Callable[[np.ndarray], np.ndarray],
    theta0: float = 1.0e-5,
    h: float = 1.4,
) -> np.ndarray:
    """Forward solve of the Karman momentum equation on ``x_grid``.

    Parameters
    ----------
    x_grid : strictly increasing streamwise stations [m], x_grid[0] > 0
    ue_func : vectorised Ue(x) [m/s]
    cf_func : vectorised local Cf(x) (e.g. ``lambda x: cf_local(x, v, ks(x))``)
    theta0 : initial momentum thickness at x_grid[0] [m]
    h : constant shape parameter

    Returns
    -------
    theta(x) on ``x_grid`` [m]
    """
    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1 or x_grid.size < 2 or not np.all(np.diff(x_grid) > 0):
        raise ValueError("x_grid must be a strictly increasing 1D grid")
    if x_grid[0] <= 0.0:
        raise ValueError("x_grid must start at x > 0")

    def rhs(x: float, y: np.ndarray) -> list[float]:
        x_arr = np.array([x])
        ue = float(ue_func(x_arr)[0])
        # dUe/dx by central difference on a local stencil
        dx = max(1e-3 * x, 1e-6)
        due = float((ue_func(np.array([x + dx]))[0] - ue_func(np.array([x - dx]))[0]) / (2.0 * dx))
        cf = float(cf_func(x_arr)[0])
        dtheta = cf / 2.0 - (2.0 + h) * (y[0] / ue) * due
        return [dtheta]

    sol = solve_ivp(
        rhs,
        t_span=(x_grid[0], x_grid[-1]),
        y0=[theta0],
        t_eval=x_grid,
        method="RK45",
        rtol=1e-8,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Karman ODE solve failed: {sol.message}")
    return sol.y[0]
