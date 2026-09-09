"""Tests for added wave resistance model."""

import numpy as np
import pytest

from hulltwin.physics.waves import added_resistance_aw, heading_factor


def test_zero_in_calm_water():
    res = added_resistance_aw(hs=0.0, beam=32.0, ship_length=200.0)
    assert res == 0.0


def test_scales_with_hs_squared():
    r1 = added_resistance_aw(hs=1.0, beam=32.0, ship_length=200.0)
    r2 = added_resistance_aw(hs=2.0, beam=32.0, ship_length=200.0)
    assert r2 == pytest.approx(4.0 * r1)


def test_head_seas_worse_than_following():
    r_head = added_resistance_aw(hs=3.0, beam=32.0, ship_length=200.0, mu_deg=0.0)
    r_follow = added_resistance_aw(hs=3.0, beam=32.0, ship_length=200.0, mu_deg=170.0)
    assert r_head > r_follow


def test_heading_factor_bounds():
    f = heading_factor(np.linspace(0, 180, 50))
    assert np.all(f >= 0.2)
    assert np.all(f <= 1.0)


def test_magnitude_order():
    # Typical bulk carrier in Hs=4 m head seas: added resistance O(1e5) N
    res = added_resistance_aw(hs=4.0, beam=32.0, ship_length=200.0, c_aw=0.5)
    assert 1e4 < res < 5e5
