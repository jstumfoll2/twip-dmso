"""Reproduce the DMSO vs. Kalman simulation figures (``main_bala_discrete.m``).

    uv run python scripts/legacy/run_dmso_sim.py noise_with_uncertainty
    uv run python scripts/legacy/run_dmso_sim.py no_noise_with_uncertainty --steps 5000 --out figs/
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from twip.legacy import plots
from twip.legacy.sim_dmso import preset, run

PRESETS = ["no_noise_no_uncertainty", "no_noise_with_uncertainty", "noise_with_uncertainty", "noise_with_uncertainty_perturbed"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=PRESETS)
    ap.add_argument("--steps", type=int, help="override number of time steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="save PNGs here instead of showing windows")
    args = ap.parse_args()

    cfg = replace(preset(args.preset), seed=args.seed)
    if args.steps:
        cfg = replace(cfg, steps=args.steps)
    r = run(cfg)
    print(f"Klqr = {r.Klqr}")

    figs = {
        "control": plots.control(r),
        "uncertainty": plots.dmso_uncertainty(r),
        "states": plots.dmso_states(r),
        "errors": plots.dmso_states(r, errors=True),
        "errors_log": plots.dmso_states(r, errors=True, log=True),
    }
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(args.out / f"{args.preset}_{name}.png", dpi=120)
        print(f"saved {len(figs)} figures to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
