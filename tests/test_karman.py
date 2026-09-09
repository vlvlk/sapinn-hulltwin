"""Tests for the Karman momentum integral solver."""

import numpy as np

from hulltwin.physics.friction import cf_local
from hulltwin.physics.karman import karman_residual, solve_momentum_boundary_layer


def test_residual_zero_for_consistent_fields():
    # Flat plate, constant Ue: exact discrete solution with piecewise-constant
    # dtheta/dx = cf(x_{i+1})/2 per interval -> residual identically zero.
    x = np.linspace(1.0, 100.0, 200)
    v = 10.0
    cf = cf_local(x, v, ks=0.0)
    theta = np.concatenate([[1e-5], 1e-5 + np.cumsum(cf[1:] / 2.0 * np.diff(x))])
    res = karman_residual(theta[1:], cf[1:] / 2.0, ue=10.0, due_dx=0.0, cf=cf[1:])
    np.testing.assert_allclose(res, 0.0, atol=1e-14)


def test_residual_raises_on_negative_ue():
    import pytest

    with pytest.raises(ValueError):
        karman_residual(1e-5, 0.0, ue=-1.0, due_dx=0.0, cf=0.003)


def test_flat_plate_growth_matches_quadrature():
    # With constant Ue and known Cf(x), theta(x) = theta0 + integral of Cf/2 dx.
    x = np.linspace(0.5, 60.0, 400)
    v = 8.0
    cf_vals = cf_local(x, v, ks=0.0)

    def cf_func(arr):
        return cf_local(arr, v, ks=0.0)

    theta = solve_momentum_boundary_layer(x, ue_func=lambda a: np.full_like(a, v), cf_func=cf_func)
    quad = np.concatenate([[0.0], np.cumsum(0.5 * (cf_vals[1:] + cf_vals[:-1]) / 2.0 * np.diff(x))])
    np.testing.assert_allclose(theta, theta[0] + quad, rtol=5e-3)


def test_theta_monotonic_growth_flat_plate():
    x = np.linspace(0.5, 100.0, 300)
    theta = solve_momentum_boundary_layer(
        x,
        ue_func=lambda a: np.full_like(a, 12.0),
        cf_func=lambda arr: cf_local(arr, 12.0, ks=100e-6),
    )
    assert np.all(np.diff(theta) > 0)
    # Sanity of magnitude: turbulent momentum thickness over ~100 m at Cf~2e-3
    # gives theta ~ O(0.05-0.15 m)
    assert 1e-4 < theta[-1] < 0.3
