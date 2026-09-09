"""Synthetic voyage generator: a "ground-truth" ship performing daily records.

The generator plays the role of reality against which the PINN observer is
validated (roadmap stage 0: identifiability of k_s from sparse logs):

- hull fouling: k_s(x, t) grows in time, is reset by cleaning events, and is
  heterogeneous along the hull (stern fouls faster than the bow);
- propeller degradation: torque penalty k_p(t) >= 1 grows monotonically
  (blade fouling/erosion) — independent of the hull;
- environment: speed, significant wave height and heading sampled daily;
- sensors: shaft power, torque, RPM, speed and fuel flow with Gaussian noise;
  this mimics noon-report / low-frequency log quality.

All physics is delegated to :mod:`hulltwin.physics` so the generator and the
observer share exactly the same model family (hybrid-PINN premise).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from hulltwin.physics.friction import NU_SEAWATER, cf_ittc57, roughness_allowance
from hulltwin.physics.propeller import PropellerParams, kq, kt
from hulltwin.physics.waves import added_resistance_aw


@dataclass(frozen=True)
class SyntheticVessel:
    """Main parameters of the simulated vessel (defaults ~ handysize bulk carrier)."""

    length: float = 180.0  # LBP [m]
    beam: float = 30.0  # [m]
    wetted_surface: float = 5300.0  # [m^2]
    form_factor: float = 0.2  # (1+k) form correction on friction
    wake_fraction: float = 0.25  # Taylor wake fraction w
    thrust_deduction: float = 0.15  # thrust deduction t_d (reserved, MVP unused)
    eta_shaft: float = 0.98  # shaft transmission efficiency
    sfoc: float = 185.0  # specific fuel oil consumption [g/kWh]
    rho: float = 1025.0  # [kg/m^3]
    propeller: PropellerParams = field(default_factory=PropellerParams)


def ks_field(
    x_norm: np.ndarray,
    age_days: float,
    base_ks: float = 30e-6,
    growth_rate: float = 0.9e-6,  # [m/day] of uniform fouling
    stern_bias: float = 0.6,
) -> np.ndarray:
    """True roughness field k_s(x, t): base + growth * age, stern-weighted.

    Parameters
    ----------
    x_norm : normalised streamwise coordinate(s) in [0, 1] (0 = FP, 1 = AP)
    age_days : days since the last hull cleaning
    base_ks : roughness right after cleaning [m]
    growth_rate : linear fouling growth [m/day]
    stern_bias : amplitude of the linear stern-forward roughness gradient
    """
    x_norm = np.asarray(x_norm, dtype=float)
    if np.any(x_norm < 0.0) or np.any(x_norm > 1.0):
        raise ValueError("x_norm must be within [0, 1]")
    spatial = 1.0 + stern_bias * (x_norm - 0.5)  # stern (x=1) rougher than bow
    return (base_ks + growth_rate * age_days) * spatial


def ks_mean_daily(x_norm: np.ndarray, age_days: float, **kwargs: float) -> float:
    """Cf-weighted mean k_s over the hull — the quantity the ship-scale
    friction integral actually sees (Cf-weighting approximated by uniform)."""
    return float(np.mean(ks_field(x_norm, age_days, **kwargs)))


def total_resistance(
    v: float,
    hs: float,
    mu_deg: float,
    ks_mean: float,
    vessel: SyntheticVessel,
) -> float:
    """Calm-water + wave resistance [N] for a Cf-weighted mean roughness."""
    re = v * vessel.length / NU_SEAWATER
    cf = float(cf_ittc57(re)) + float(roughness_allowance(ks_mean, vessel.length))
    r_f = 0.5 * vessel.rho * v**2 * vessel.wetted_surface * cf
    r_form = vessel.form_factor * r_f
    r_aw = float(added_resistance_aw(hs, vessel.beam, vessel.length, rho=vessel.rho, mu_deg=mu_deg))
    return r_f + r_form + r_aw


def solve_shaft_rate(v: float, r_t: float, vessel: SyntheticVessel) -> float:
    """Solve self-propulsion: thrust identity rho n^2 D^4 K_T(J) = (1-t_d) R_T,
    with J = (1-w) V / (n D). Returns shaft revolution rate n [rev/s]."""
    p = vessel.propeller
    a = (1.0 - vessel.wake_fraction) * v / p.diameter  # n = a / J

    def thrust_mismatch(j: float) -> float:
        n = a / j
        if n <= 0:
            return -r_t
        kt_v = float(kt(j, p))
        if kt_v <= 0:
            return -r_t
        return p.rho * n**2 * p.diameter**4 * kt_v * (1.0 - vessel.thrust_deduction) - r_t

    j_lo, j_hi = 0.05, 0.95
    lo, hi = thrust_mismatch(j_lo), thrust_mismatch(j_hi)
    if lo * hi > 0:
        raise RuntimeError(
            f"No self-propulsion point in J in [{j_lo}, {j_hi}] for V={v:.2f} m/s, R_T={r_t:.0f} N"
        )
    j_sol = brentq(thrust_mismatch, j_lo, j_hi, xtol=1e-8)
    return a / j_sol


def simulate_voyage(
    n_days: int = 720,
    seed: int = 42,
    cleanings: tuple[int, ...] = (200, 430),
    base_ks: float = 30e-6,
    growth_rate: float = 0.9e-6,
    kp_growth: float = 3.5e-4,  # propeller degradation per day [1/day]
    noise_power: float = 0.02,
    noise_torque: float = 0.02,
    noise_speed: float = 0.05,
    vessel: SyntheticVessel | None = None,
) -> pd.DataFrame:
    """Simulate daily log records over ``n_days``.

    Returns a DataFrame with one row per day containing measured signals
    (noisy) and ground-truth hidden states (``ks_mean_true``, ``kp_true``,
    ``days_since_cleaning``).
    """
    vessel = vessel or SyntheticVessel()
    rng = np.random.default_rng(seed)
    x_norm = np.linspace(0.0, 1.0, 101)
    cleanings_set = set(cleanings)

    rows: list[dict[str, float]] = []
    age = 0.0
    for day in range(n_days):
        if day in cleanings_set:
            age = 0.0
        kp_true = 1.0 + kp_growth * day

        v = float(rng.uniform(5.5, 7.5))  # ~11-15 kn
        hs = float(rng.uniform(0.2, 5.0))
        mu_deg = float(rng.uniform(0.0, 180.0))

        ks_mean = ks_mean_daily(x_norm, age, base_ks=base_ks, growth_rate=growth_rate)
        r_t = total_resistance(v, hs, mu_deg, ks_mean, vessel)
        n = solve_shaft_rate(v, r_t, vessel)

        p = vessel.propeller
        j = (1.0 - vessel.wake_fraction) * v / (n * p.diameter)
        q = float(p.rho * n**2 * p.diameter**5 * kq(j, p, kp=kp_true))
        p_shaft = 2.0 * np.pi * n * q / vessel.eta_shaft  # [W]
        fuel_t_day = p_shaft / 1000.0 * 24.0 * vessel.sfoc / 1000.0  # [t/day]

        rows.append(
            {
                "day": float(day),
                "v_meas": v + float(rng.normal(0.0, noise_speed)),
                "hs": hs,
                "wave_heading_deg": mu_deg,
                "rpm_meas": n * 60.0 * float(rng.normal(1.0, noise_torque / 3.0)),
                "torque_meas": q * float(rng.normal(1.0, noise_torque)),
                "power_meas": p_shaft * float(rng.normal(1.0, noise_power)),
                "fuel_t_day_meas": fuel_t_day * float(rng.normal(1.0, noise_power)),
                # ground truth (hidden in real logs)
                "ks_mean_true": ks_mean,
                "kp_true": kp_true,
                "days_since_cleaning": age,
                "cleaning_today": float(day in cleanings_set),
            }
        )
        age += 1.0

    return pd.DataFrame(rows)
