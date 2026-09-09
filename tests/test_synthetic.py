"""Tests for the synthetic voyage generator (ground-truth sanity)."""

import numpy as np
import pandas as pd
import pytest

from hulltwin.data.synthetic import (
    SyntheticVessel,
    ks_field,
    simulate_voyage,
    solve_shaft_rate,
    total_resistance,
)
from hulltwin.physics.propeller import kq


@pytest.fixture(scope="module")
def voyage() -> pd.DataFrame:
    return simulate_voyage(n_days=400, cleanings=(200,), seed=7)


def test_schema_and_completeness(voyage: pd.DataFrame):
    expected = {
        "day",
        "v_meas",
        "hs",
        "wave_heading_deg",
        "rpm_meas",
        "torque_meas",
        "power_meas",
        "fuel_t_day_meas",
        "ks_mean_true",
        "kp_true",
        "days_since_cleaning",
        "cleaning_today",
    }
    assert expected <= set(voyage.columns)
    assert len(voyage) == 400
    assert not voyage.isna().any().any()


def test_cleaning_resets_fouling(voyage: pd.DataFrame):
    before = voyage.loc[195:199, "ks_mean_true"].mean()  # day ~200, aged hull
    after = voyage.loc[201:205, "ks_mean_true"].mean()  # fresh hull
    assert after < 0.5 * before
    assert voyage["days_since_cleaning"].iloc[205] == 5.0


def test_fouling_degrades_power_at_fixed_conditions():
    """Same V, calm water: power grows with fouling age."""
    vessel = SyntheticVessel()
    v, hs, mu = 6.5, 0.0, 0.0
    p_fresh = total_resistance(v, hs, mu, ks_field(np.array([0.5]), 0.0)[0], vessel)
    p_fouled = total_resistance(v, hs, mu, ks_field(np.array([0.5]), 400.0)[0], vessel)
    assert p_fouled > p_fresh
    # +10..30% friction => here a few-to-tens of percent on total resistance
    ratio = p_fouled / p_fresh
    assert 1.01 < ratio < 1.5


def test_power_increases_with_wave_height():
    vessel = SyntheticVessel()
    v, ks = 6.5, 60e-6
    r_calm = total_resistance(v, 0.0, 0.0, ks, vessel)
    r_storm = total_resistance(v, 5.0, 0.0, ks, vessel)
    assert r_storm > r_calm


def test_torque_identity_with_true_kp(voyage: pd.DataFrame):
    """Generated torque must satisfy the K_Q identity with the true degradation."""
    p = SyntheticVessel().propeller
    df = voyage
    n = df["rpm_meas"] / 60.0
    v = df["v_meas"]
    j = (1.0 - SyntheticVessel().wake_fraction) * v / (n * p.diameter)
    q_hat = (
        p.rho
        * n**2
        * p.diameter**5
        * np.array([float(kq(jj, p, kp=kpp)) for jj, kpp in zip(j, df["kp_true"], strict=True)])
    )
    # Noisy rpm/v feed the J recomputation (~3% 1-sigma on q_hat), plus 2%
    # measurement noise on torque itself; the identity is checked on the bulk
    # (median) and a wide tail tolerance.
    rel = np.abs(q_hat - df["torque_meas"]) / df["torque_meas"]
    assert np.median(rel) < 0.03
    assert rel.max() < 0.12


def test_fuel_tracks_power(voyage: pd.DataFrame):
    corr = np.corrcoef(voyage["power_meas"], voyage["fuel_t_day_meas"])[0, 1]
    assert corr > 0.95


def test_shaft_rate_positive_and_reasonable():
    vessel = SyntheticVessel()
    r_t = total_resistance(6.5, 1.0, 30.0, 50e-6, vessel)
    n = solve_shaft_rate(6.5, r_t, vessel)
    # Typical bulk carrier: 90-150 rpm ~ 1.5-2.5 rev/s
    assert 1.0 < n < 3.5
