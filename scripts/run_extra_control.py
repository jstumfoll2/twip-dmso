"""Run the neural-network extra-control simulation (``ExtraControl_v5.m``).

    uv run python scripts/run_extra_control.py unmodeled_dynamics
    uv run python scripts/run_extra_control.py deadzone_and_backlash --out figs/
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from twip import plots
from twip.extra_control import preset, run

PRESETS = ["as_saved", "unmodeled_dynamics", "parameter_uncertainty", "deadzone", "deadzone_and_backlash"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=PRESETS)
    ap.add_argument("--steps", type=int, help="override number of time steps")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random NN input layers (and noise)")
    ap.add_argument("--out", type=Path, help="save PNGs here instead of showing windows")
    args = ap.parse_args()

    cfg = replace(preset(args.preset), seed=args.seed)
    if args.steps:
        cfg = replace(cfg, steps=args.steps)
    r = run(cfg)
    print(f"LQR error {r.error_lqr:.1f} | extra control error {r.error_extra:.1f} | nominal error {r.error_nom:.1f}")

    figs = {
        "states": plots.extra_control_states(r),
        "control": plots.extra_control_inputs(r),
        "uncertainty": plots.extra_control_uncertainty(r),
    }
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(args.out / f"extra_{args.preset}_{name}.png", dpi=120)
        print(f"saved {len(figs)} figures to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
