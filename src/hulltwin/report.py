"""End-to-end MVP demo report: simulate -> observe -> plot.

Produces a single multi-panel figure (``reports/hulltwin_report.png``)
showing the identified hidden states against ground truth:
    1. k_s_mean(t): hull fouling trajectory, cleaning events marked;
    2. k_p(t): propeller degradation;
    3. k_s(x, t) field heatmap (prior-based spatial shape, diagnostic);
    4. power fit: measured vs reconstructed shaft power.

Run: ``uv run python scripts/generate_report.py [--days N --epochs E]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hulltwin.data.synthetic import simulate_voyage
from hulltwin.pinn.observer import HullTwinObserver


def _power_reconstruction(observer: HullTwinObserver, df: pd.DataFrame) -> np.ndarray:
    """Reconstructed shaft power from identified states + measured rpm/speed."""
    batch = {
        "v": torch.tensor(df["v_meas"].to_numpy(), dtype=torch.float32),
        "n": torch.tensor(df["rpm_meas"].to_numpy() / 60.0, dtype=torch.float32),
        "hs": torch.tensor(df["hs"].to_numpy(), dtype=torch.float32),
        "mu_rad": torch.tensor(np.radians(df["wave_heading_deg"].to_numpy()), dtype=torch.float32),
        "torque": torch.tensor(df["torque_meas"].to_numpy(), dtype=torch.float32),
        "day": torch.tensor(np.arange(len(df)), dtype=torch.long),
        "cleaning": torch.tensor(df["cleaning_today"].to_numpy() > 0.5),
    }
    with torch.no_grad():
        res = observer.physics_residuals(batch)
        # P_hat = P_via_torque_identity * (1 + thrust misfit applied to hull part)
        p = observer.vessel.propeller
        n = batch["n"]
        j = (1.0 - observer.vessel.wake_fraction) * batch["v"] / (n * p.diameter)
        kq_c = torch.tensor(p.kq_coeffs, dtype=torch.float32)
        kq_v = (kq_c[0] + kq_c[1] * j + kq_c[2] * j * j) * observer.kp(batch["day"])
        q_hat = p.rho * n**2 * p.diameter**5 * kq_v
        p_hat = 2.0 * np.pi * n * q_hat / observer.vessel.eta_shaft
        # correct for the hull-side misfit (thrust residual) so that the plot
        # shows the full model reconstruction, not just the torque leg
        return (p_hat * (1.0 + res["thrust"])).detach().numpy()


def build_report(
    df: pd.DataFrame,
    observer: HullTwinObserver,
    out_path: Path,
    fit_history: list[float] | None = None,
) -> dict[str, float]:
    """Render the report figure; returns summary metrics."""
    days = np.arange(len(df))
    ks_pred = observer.predict_ks_mean(days)
    ks_true = df["ks_mean_true"].to_numpy() * 1e6
    kp_pred = observer.predict_kp(days)
    kp_true = df["kp_true"].to_numpy()
    p_meas = df["power_meas"].to_numpy() / 1e6  # MW
    p_hat = _power_reconstruction(observer, df) / 1e6

    corr_ks = float(np.corrcoef(ks_pred, ks_true)[0, 1])
    err_kp = float(abs(kp_pred[-1] - kp_true[-1]) / kp_true[-1])
    err_p = float(np.median(np.abs(p_hat - p_meas) / p_meas))

    cleanings = df.loc[df["cleaning_today"] > 0.5, "day"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    ax.plot(days, ks_true, "k-", lw=2, label="истина $\\bar{k}_s(t)$")
    ax.plot(days, ks_pred, "r--", lw=1.5, label="PINN-оценка")
    for c in cleanings:
        ax.axvline(c, color="tab:blue", ls=":", alpha=0.7)
    ax.set_title(f"Обрастание корпуса: corr = {corr_ks:.3f} (|: очистки)")
    ax.set_xlabel("день")
    ax.set_ylabel("$\\bar{k}_s$, мкм")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(days, kp_true, "k-", lw=2, label="истина $k_p(t)$")
    ax.plot(days, kp_pred, "r--", lw=1.5, label="PINN-оценка")
    ax.set_title(f"Деградация винта: ошибка в конце = {err_kp:.1%}")
    ax.set_xlabel("день")
    ax.set_ylabel("$k_p$ (штраф момента)")
    ax.legend()

    ax = axes[1, 0]
    field = observer.predict_ks_field(days, n_x=101)
    im = ax.imshow(
        field,
        aspect="auto",
        origin="lower",
        extent=[days[0], days[-1], 0, 1],
        cmap="turbo",
    )
    fig.colorbar(im, ax=ax, label="$k_s$, мкм")
    ax.set_title("Поле $k_s(x,t)$ (форма — априорная, см. docstring)")
    ax.set_xlabel("день")
    ax.set_ylabel("x / L (0 — нос, 1 — корма)")

    ax = axes[1, 1]
    ax.plot(days, p_meas, "k-", lw=1, label="P измеренная")
    ax.plot(days, p_hat, "r--", lw=1.2, label="P модель")
    ax.set_title(f"Мощность на валу: медианная ошибка = {err_p:.1%}")
    ax.set_xlabel("день")
    ax.set_ylabel("P, МВт")
    ax.legend()

    if fit_history:
        print(f"loss: {fit_history[0]:.4f} -> {fit_history[-1]:.6f}")
    fig.suptitle("PINN-HullTwin: MVP-отчёт (синтетика)", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    return {"ks_corr": corr_ks, "kp_end_err": err_kp, "power_median_err": err_p}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PINN-HullTwin MVP report")
    parser.add_argument("--days", type=int, default=720)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--cleanings", type=int, nargs="*", default=[200, 430])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("reports/hulltwin_report.png"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = simulate_voyage(n_days=args.days, cleanings=tuple(args.cleanings), seed=args.seed)
    observer = HullTwinObserver(n_days=args.days)
    history = observer.fit(df, epochs=args.epochs, lr=5e-2)
    metrics = build_report(df, observer, args.out, history)
    print(f"report saved to {args.out}")
    print(f"metrics: {metrics}")


if __name__ == "__main__":
    main()
