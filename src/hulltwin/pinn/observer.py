"""PINN observer v0 (hybrid): recover k_s(t) and propeller degradation k_p(t).

Identifiability analysis (roadmap step 5, done up-front) shows that with
ship-scale daily logs the data contain exactly ONE equation per day for the
hull (the thrust identity) — i.e. only the Cf-weighted mean roughness
k_s_mean(t) is identifiable, not the spatial distribution k_s(x). Trying to
fit a free (x, t) MLP from integral data alone is ill-posed (optimiser
collapses to a constant field). Therefore the observer is structured in two
stages:

- Stage 1 (data-driven, well-posed): per-day hidden state
    * k_s_day(t)  >= 0   — hull mean roughness, one free parameter per day,
      coupled by a relative-smoothness penalty (except across cleaning days);
    * k_p(t) = 1 + softplus(rate) * t — propeller torque penalty.
  Physics residuals (same correlations as :mod:`hulltwin.physics`):
    * thrust identity: rho n^2 D^4 K_T(J) (1 - t_d) = R_T(V, Hs, mu, ks)
      -> driven by hull roughness (ks enters through Delta CF);
    * torque identity: Q_meas = rho n^2 D^5 K_Q(J) k_p
      -> separates propeller wear from hull fouling.
  J and n come from measured RPM and log speed.

- Stage 2 (prior-informed): the spatial field is reconstructed as
    k_s(x, t) = k_s_day(t) * (1 + beta * (x_norm - 1/2)),
  with beta = stern fouling gradient taken from prior knowledge (ROV
  inspections of sister ships). It is a DIAGNOSTIC, not a measurement:
  strip-level (e.g. IMU/vibration or local heat-flux) data would be needed
  to identify beta — flagged for the stage-2 upgrade of the roadmap.

Loss = w_thrust * r_thrust^2 + w_torque * r_torque^2 + w_dt * (Delta ks/ks)^2
(all residuals dimensionless).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

from hulltwin.data.synthetic import SyntheticVessel


class HullTwinObserver(nn.Module):
    """Differentiable twin: k_s_day(t) + k_p(t) + ship-scale thrust/torque physics."""

    def __init__(
        self,
        vessel: SyntheticVessel | None = None,
        n_days: int = 0,
        stern_bias_prior: float = 0.6,
    ) -> None:
        super().__init__()
        self.vessel = vessel or SyntheticVessel()
        self.n_days = n_days
        self.stern_bias_prior = stern_bias_prior
        # stage-1 hidden states (created for n_days; zeros -> softplus ~ 0.69 um,
        # overwritten during fit)
        self.ks_day_raw = nn.Parameter(torch.full((max(n_days, 1),), np.log(np.expm1(50.0))))
        self.log_kp_rate = nn.Parameter(torch.tensor(np.log(np.expm1(3.0e-4)), dtype=torch.float32))

    # ------------------------------------------------------------------ #
    # hidden states
    # ------------------------------------------------------------------ #
    def ks_day(self, day_idx: Tensor) -> Tensor:
        """Mean hull roughness per day [m]; day_idx: long tensor of day numbers."""
        return nn.functional.softplus(self.ks_day_raw[day_idx.long()]) * 1e-6

    def kp(self, day_idx: Tensor) -> Tensor:
        """Propeller torque penalty k_p(t) >= 1."""
        t = day_idx.long().to(torch.float32)
        return 1.0 + nn.functional.softplus(self.log_kp_rate) * t

    def ks_field(self, day_idx: Tensor, n_x: int = 101) -> Tensor:
        """Stage-2 spatial reconstruction k_s(x, t) [m], shape (n_x, nd).

        PRIOR-BASED shape (stern fouls faster); not identified from
        integral ship data — see module docstring.
        """
        x = torch.linspace(0.0, 1.0, n_x, device=day_idx.device)
        shape = 1.0 + self.stern_bias_prior * (x - 0.5)  # [nx]
        ks = self.ks_day(day_idx)  # [nd]
        return shape.unsqueeze(1) * ks.unsqueeze(0)

    # ------------------------------------------------------------------ #
    # differentiable physics (torch mirror of hulltwin.physics)
    # ------------------------------------------------------------------ #
    def physics_residuals(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        v = batch["v"]
        n = batch["n"]  # rev/s from measured rpm
        hs = batch["hs"]
        mu = batch["mu_rad"]
        day_idx = batch["day"]
        q_meas = batch["torque"]

        p = self.vessel.propeller
        dev = v.device
        kt_c = torch.tensor(p.kt_coeffs, dtype=torch.float32, device=dev)
        kq_c = torch.tensor(p.kq_coeffs, dtype=torch.float32, device=dev)
        j = (1.0 - self.vessel.wake_fraction) * v / (n * p.diameter)

        kt_v = kt_c[0] + kt_c[1] * j + kt_c[2] * j * j
        kq_v = (kq_c[0] + kq_c[1] * j + kq_c[2] * j * j) * self.kp(day_idx)

        # thrust identity with the ks-dependent resistance
        ks_m = self.ks_day(day_idx)
        re = v * self.vessel.length / 1.05e-6
        log_re = torch.log10(re)
        cf0 = 0.075 / (log_re - 2.0) ** 2
        dcf = (0.105 * (ks_m / self.vessel.length) ** (1.0 / 3.0) - 0.64e-3).clamp(min=0.0)
        cf = cf0 + dcf
        r_f = 0.5 * self.vessel.rho * v**2 * self.vessel.wetted_surface * cf
        r_form = self.vessel.form_factor * r_f
        mu_c = torch.clip(0.6 + 0.4 * torch.cos(mu), 0.2, 1.0)
        r_aw = (
            0.5
            * self.vessel.rho
            * 9.81
            * (hs / 2.0) ** 2
            * self.vessel.beam**2
            / self.vessel.length
        ) * mu_c
        r_t = r_f + r_form + r_aw
        thrust = self.vessel.rho * n**2 * p.diameter**4 * kt_v
        r_thrust = (thrust * (1.0 - self.vessel.thrust_deduction) - r_t) / r_t

        # torque identity
        q_hat = self.vessel.rho * n**2 * p.diameter**5 * kq_v
        r_torque = (q_meas - q_hat) / q_hat

        return {"thrust": r_thrust, "torque": r_torque}

    # ------------------------------------------------------------------ #
    # loss / training
    # ------------------------------------------------------------------ #
    def loss(self, batch: dict[str, Tensor], w_dt: float = 2.0) -> Tensor:
        res = self.physics_residuals(batch)
        ks = self.ks_day(batch["day"])
        cleaning = batch["cleaning"].bool()
        jump_mask = ~(cleaning[:-1] | cleaning[1:])
        rel_jumps = ((ks[1:] - ks[:-1]) / (ks[:-1] + 1e-6))[jump_mask]
        smooth = rel_jumps.pow(2).mean() if rel_jumps.numel() else ks.sum() * 0.0
        return 1.0 * res["thrust"].pow(2).mean() + 1.0 * res["torque"].pow(2).mean() + w_dt * smooth

    def _make_batch(self, df: pd.DataFrame, stride: int) -> dict[str, Tensor]:
        idx = np.arange(0, len(df), stride)
        return {
            "v": torch.tensor(df["v_meas"].to_numpy()[idx], dtype=torch.float32),
            "n": torch.tensor(df["rpm_meas"].to_numpy()[idx] / 60.0, dtype=torch.float32),
            "hs": torch.tensor(df["hs"].to_numpy()[idx], dtype=torch.float32),
            "mu_rad": torch.tensor(
                np.radians(df["wave_heading_deg"].to_numpy()[idx]), dtype=torch.float32
            ),
            "torque": torch.tensor(df["torque_meas"].to_numpy()[idx], dtype=torch.float32),
            "day": torch.tensor(idx, dtype=torch.long),
            "cleaning": torch.tensor(df["cleaning_today"].to_numpy()[idx] > 0.5),
        }

    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 3000,
        lr: float = 5e-2,
        stride: int = 1,  # kept for API compatibility; stage-1 is cheap — use 1
        verbose_every: int = 0,
    ) -> list[float]:
        """Train stage-1 hidden states on a daily-log DataFrame.

        (see :func:`hulltwin.data.synthetic.simulate_voyage` for the schema).
        """
        if len(df) != self.n_days:
            # (re)allocate per-day parameters to match the log
            self.ks_day_raw = nn.Parameter(
                torch.full((len(df),), float(self.ks_day_raw.detach().mean()))
            )
            self.n_days = len(df)
        batch = self._make_batch(df, stride)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        history: list[float] = []
        for epoch in range(epochs):
            opt.zero_grad()
            loss = self.loss(batch)
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))
            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"epoch {epoch + 1}: loss={history[-1]:.6f}")
        return history

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_ks_mean(self, days: np.ndarray) -> np.ndarray:
        day = torch.tensor(np.asarray(days, dtype=float), dtype=torch.long)
        return self.ks_day(day).numpy() * 1e6  # -> um

    @torch.no_grad()
    def predict_kp(self, days: np.ndarray) -> np.ndarray:
        day = torch.tensor(np.asarray(days, dtype=float), dtype=torch.long)
        return self.kp(day).numpy()

    @torch.no_grad()
    def predict_ks_field(self, days: np.ndarray, n_x: int = 101) -> np.ndarray:
        """Prior-shaped k_s(x, t) [um], shape (n_x, len(days)) — diagnostic only."""
        day = torch.tensor(np.asarray(days, dtype=float), dtype=torch.long)
        return self.ks_field(day, n_x).numpy() * 1e6
