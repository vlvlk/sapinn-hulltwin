"""Tests for friction correlations against known analytic values."""

import numpy as np
import pytest

from hulltwin.physics.friction import cf_ittc57, cf_local, roughness_allowance


def test_ittc57_known_value():
    # C_F0(Re = 1e8) = 0.075 / (8 - 2)^2 = 0.00208(3)
    assert cf_ittc57(1e8) == pytest.approx(0.075 / 36.0, rel=1e-12)


def test_ittc57_decreases_with_re():
    res = cf_ittc57(np.array([1e6, 1e7, 1e8, 1e9]))
    assert np.all(np.diff(res) < 0)


def test_ittc57_invalid_re():
    with pytest.raises(ValueError):
        cf_ittc57(10.0)


def test_roughness_allowance_typical_value():
    # ks = 150 um on L = 200 m -> (ks/L)^(1/3) ~ 9.08e-3
    # 0.105 * 9.08e-3 - 0.64e-3 ~ 3.1e-4 (typical painted-hull allowance)
    assert roughness_allowance(150e-6, 200.0) == pytest.approx(3.1e-4, rel=0.05)


def test_roughness_allowance_smooth_clamped_to_zero():
    assert roughness_allowance(0.0, 200.0) == 0.0
    # ultra-smooth surface: allowance below 0 -> clamped
    assert roughness_allowance(1e-9, 5.0) == 0.0


def test_roughness_allowance_monotonic_in_ks():
    ks = np.linspace(0.0, 1e-3, 50)
    res = roughness_allowance(ks, 150.0)
    assert np.all(np.diff(res) >= 0)


def test_cf_local_smooth_matches_ittc_shape():
    v = 10.0
    x = np.array([10.0, 50.0, 150.0])
    np.testing.assert_allclose(cf_local(x, v, ks=0.0), cf_ittc57(v * x / 1.05e-6))


def test_cf_local_roughness_increases_friction():
    x = np.full(20, 100.0)
    ks = np.linspace(0.0, 500e-6, 20)
    res = cf_local(x, 8.0, ks=ks)
    assert np.all(np.diff(res) >= 0)
    assert res[-1] > res[0] * 1.1  # noticeable penalty at heavy fouling
