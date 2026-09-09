"""Tests for propeller open-water model and torque identity."""

import numpy as np
import pytest

from hulltwin.physics.propeller import (
    PropellerParams,
    kq,
    kt,
    open_water_efficiency,
    torque_residual,
)


def test_kt_kq_decrease_with_j():
    j = np.linspace(0.05, 0.9, 50)
    assert np.all(np.diff(kt(j)) < 0)
    assert np.all(np.diff(kq(j)) < 0)


def test_efficiency_physical_range():
    # Quadratic approximation of B4-70 peaks near eta0 ~ 0.9 at J ~ 0.75;
    # validity range of the polynomial is J <= 0.8
    j = np.linspace(0.1, 0.8, 100)
    eta = open_water_efficiency(j)
    assert np.nanmax(eta) < 0.95
    assert np.nanmax(eta) > 0.5


def test_kq_degradation_penalty():
    j = np.full(10, 0.6)
    clean = kq(j)
    fouled = kq(j, kp=1.15)
    np.testing.assert_allclose(fouled, clean * 1.15)


def test_kq_rejects_kp_below_one():
    with pytest.raises(ValueError):
        kq(0.6, kp=0.9)


def test_torque_residual_zero_for_self_consistent_point():
    p = PropellerParams()
    n = 2.0  # rev/s
    j = 0.6
    q = p.rho * n**2 * p.diameter**5 * float(kq(j, p))
    assert torque_residual(q, n, j, params=p) == pytest.approx(0.0, abs=1e-6 * q)


def test_torque_residual_detects_fouling_at_clean_kp():
    # If blades foul (true kp = 1.2) but observer assumes clean, residual is +Q*(kp-1)
    p = PropellerParams()
    n, j, true_kp = 2.0, 0.6, 1.2
    q = p.rho * n**2 * p.diameter**5 * float(kq(j, p, kp=true_kp))
    res = torque_residual(q, n, j, params=p, kp=1.0)
    assert res > 0
    assert res == pytest.approx(q * (1.0 - 1.0 / 1.2), rel=1e-6)
