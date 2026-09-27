"""Run the corrected baseline: LQR + Kalman estimator on the corrected plant.

    uv run python scripts/run_baseline.py
    uv run python scripts/run_baseline.py --estimator complementary --tilt 2
    uv run python scripts/run_baseline.py --scale ke=0.5 --scale Mp=1.1 --out figs/
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from twip.baseline import (
    HARDWARE_DEFAULT_GAINS,
    ComplementaryEstimator,
    KalmanEstimator,
    LQRController,
    SimConfig,
    SimResult,
    design_lqr,
    simulate,
)
from twip.params import CORRECTED

R2D = 180 / np.pi


def plot(r: SimResult, title: str) -> list[plt.Figure]:
    t = r.time
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (k, s, name, unit) in zip(
        axs.flat, [(2, R2D, "Tilt", "deg"), (3, R2D, "Tilt rate", "deg/s"), (0, 1, "Position", "m"), (1, 1, "Velocity", "m/s")]
    ):
        ax.plot(t, r.x[k] * s, label="true")
        ax.plot(t, r.xhat[k] * s, "--", label="estimate")
        ax.set(title=name, ylabel=unit)
        ax.grid(True)
    axs[0, 0].legend()
    for ax in axs[1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()

    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs2[0].plot(t[:-1], r.u_cmd, label="controller output")
    axs2[0].plot(t[:-1], r.v_applied, label="applied")
    axs2[0].set(ylabel="V", title="Control")
    axs2[0].legend()
    d = r.model_residual()
    axs2[1].plot(t[:-1], d[1], label="xdot")
    axs2[1].plot(t[:-1], d[3], label="thetadot")
    axs2[1].set(xlabel="Time (s)", title="Model residual  x[k+1] - (F x[k] + G v[k])  (what an uncertainty observer should estimate)")
    axs2[1].legend()
    for ax in axs2:
        ax.grid(True)
    fig2.tight_layout()
    return [fig, fig2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estimator", choices=["kalman", "complementary"], default="kalman")
    ap.add_argument("--gains", choices=["lqr", "hardware"], default="lqr")
    ap.add_argument("--tilt", type=float, default=5.0, help="initial tilt from rest, deg")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--delay", type=int, default=0, help="extra samples of actuation delay")
    ap.add_argument("--scale", action="append", default=[], metavar="PARAM=FACTOR", help="perturb the true plant, e.g. ke=0.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="save PNGs here instead of showing windows")
    args = ap.parse_args()

    plant = CORRECTED.scaled(**{k: float(v) for k, v in (s.split("=") for s in args.scale)}) if args.scale else CORRECTED
    cfg = SimConfig(x0=(0.0, 0.0, args.tilt / R2D, 0.0), duration=args.duration, plant=plant, delay_steps=args.delay, seed=args.seed)
    cfg = replace(cfg, sensors=replace(cfg.sensors, noise=not args.no_noise))
    K = design_lqr() if args.gains == "lqr" else HARDWARE_DEFAULT_GAINS
    est = KalmanEstimator(sensors=cfg.sensors) if args.estimator == "kalman" else ComplementaryEstimator()
    r = simulate(LQRController(K), est, cfg)

    print(f"K = {np.round(K, 4)}")
    print(f"fell: {r.fell}" + (f" at t = {r.time[-1]:.2f} s" if r.fell else ""))
    for k, v in r.metrics(settle_from=min(5.0, r.time[-1] / 2)).items():
        print(f"  {k:22s} {v:.4g}")

    figs = plot(r, f"{args.gains} gains, {args.estimator} estimator, {args.tilt:g} deg start")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(figs, 1):
            fig.savefig(args.out / f"baseline_{i}.png", dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()
