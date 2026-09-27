"""Re-run the post-processing of logged robot data.

    uv run python scripts/run_data_replays.py filters        # Tests/Filtering Test/filteringtest.m
    uv run python scripts/run_data_replays.py implementation # Thesis Programs/Implementation/lqrkalmantest1v2.m
    uv run python scripts/run_data_replays.py firmware       # robot's own DMSO (twip_v9) vs. its log
    uv run python scripts/run_data_replays.py allan          # Tests/Allan Variance/allanprocess.m (Allan part)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from twip.analysis import allan, gauss_markov_bias_variance, replay_filters, replay_firmware_v9, replay_implementation
from twip.data import IMPLEMENTATION_COLUMNS, load_log, load_mat, thesis_path

R2D = 180 / np.pi


def filters(args) -> list[plt.Figure]:
    path = args.file or thesis_path("Tests", "Filtering Test", "filteringtest12.mat")
    d = load_mat(path)
    r = replay_filters(d, firmware_bug=not args.fix_firmware_bug)
    t = r["t"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axs[0].plot(t, r["mso"], label="DMSO (Python)")
    axs[0].plot(t, r["thetak"], label="Kalman (Python)")
    axs[0].plot(t, r["thetac"], label="Complementary (Python)")
    if "pitchmso" in d:
        axs[0].plot(t, d["pitchmso"], "k:", label="DMSO (robot log)")
    axs[0].set(ylabel="Tilt (deg)", title=Path(path).name)
    axs[0].legend()
    axs[1].plot(t, r["bias"], label="Kalman gyro bias")
    axs[1].plot(t, -r["fhat1"] / d["dt"], label="-fhat1/dt (DMSO)")
    axs[1].set(xlabel="Time (s)", ylabel="deg/s")
    axs[1].legend()
    for ax in axs:
        ax.grid(True)
    fig.tight_layout()
    return [fig]


def implementation(args) -> list[plt.Figure]:
    path = args.file or thesis_path("Thesis", "Thesis Programs", "Implementation", "implementationtest5.txt")
    log = load_log(path, IMPLEMENTATION_COLUMNS)
    r = replay_implementation(log, reproduce_R_bug=args.thesis_r_bug)
    s = 310  # starttime in the MATLAB script
    n = len(r.accpitch)
    t = r.t[s:n]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, r.accpitch[s:n], label="Accelerometer")
    ax.plot(t, r.kalman[2, s:n] * R2D, label="Kalman")
    ax.plot(t, r.xhatmso[2, s:n] * R2D, label="DMSO")
    ax.plot(t, log["pitchc"][s:n], label="Complementary (robot)")
    ax.set(title=f"Tilt Angle - {Path(path).name}", xlabel="Time (s)", ylabel="deg")
    ax.legend()
    ax.grid(True)

    fig2, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(t, r.fhat[0, s:n])
    axs[0].set(title="Acceleration Uncertainty")
    axs[1].plot(t, r.fhat[1, s:n])
    axs[1].set(title="Tilt Acceleration Uncertainty", xlabel="Time (s)")
    for a in axs:
        a.grid(True)
    fig2.tight_layout()
    return [fig, fig2]


def firmware(args) -> list[plt.Figure]:
    path = args.file or thesis_path("Thesis", "Thesis Programs", "Implementation", "implementationtest5.txt")
    log = load_log(path, IMPLEMENTATION_COLUMNS)
    r = replay_firmware_v9(log)
    t = np.concatenate(([0.0], np.cumsum(log["dt"][:-1])))
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (k, title) in zip(axs.flat, [("pitchm", "Tilt (deg)"), ("gyhat", "Tilt rate (deg/s)"), ("xhat", "Position (m)"), ("xhatdot", "Velocity (m/s)")]):
        ax.plot(t, log[k], label="robot log")
        ax.plot(t, r[k], "--", label="Python FirmwareDMSO")
        ax.set(title=title)
        ax.grid(True)
    axs[0, 0].legend()
    fig.suptitle(f"On-board DMSO (twip_v9) - {Path(path).name}")
    fig.tight_layout()
    for k in ("pitchm", "gyhat", "xhat", "xhatdot"):
        print(f"max |python - robot| {k}: {np.max(np.abs(r[k] - log[k])):.2g}")
    return [fig]


def allan_plot(args) -> list[plt.Figure]:
    path = args.file or thesis_path("Tests", "Allan Variance", "allandata.mat")
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    data, tau = m["data"], m["tau"]
    r = allan(data.freq, data.rate, tau)
    print(f"gyro white-noise variance: {np.var(m['gy']):.4g} (deg/s)^2")
    print(f"gyro walking-bias variance (140-sample bins): {gauss_markov_bias_variance(m['gy']):.4g}")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(r.tau, r.osig, label="overlapping")
    ax.loglog(r.tau, r.sig2, label="non-overlapping")
    ax.set(xlabel="tau (s)", ylabel="Allan deviation", title="Gyroscope Allan deviation")
    ax.grid(True, which="both")
    ax.legend()
    return [fig]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["filters", "implementation", "firmware", "allan"])
    ap.add_argument("--file", type=Path, help="data file (defaults to the one the thesis used)")
    ap.add_argument("--fix-firmware-bug", action="store_true", help="filters: use the corrected DMSO F matrix")
    ap.add_argument("--thesis-r-bug", action="store_true", help="implementation: reproduce the thesis R overwrite")
    ap.add_argument("--out", type=Path, help="save PNGs here instead of showing windows")
    args = ap.parse_args()

    figs = {"filters": filters, "implementation": implementation, "firmware": firmware, "allan": allan_plot}[args.which](args)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(figs, 1):
            fig.savefig(args.out / f"{args.which}_{i}.png", dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()
