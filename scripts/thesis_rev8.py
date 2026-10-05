"""Generate the figures and tables for thesis revision 8 (docs/thesis).

    uv run python scripts/thesis_rev8.py            # everything
    uv run python scripts/thesis_rev8.py --only ch7 # one step: ch4, ch5, ch7, observer, control, perturbation or hardware

Writes vector PDFs to docs/thesis/figures/, LaTeX table bodies to
docs/thesis/generated/, and all numbers quoted in the text to
docs/thesis/generated/results.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twip.analysis import allan
from twip.baseline import KalmanEstimator, simulate
from twip.controllers import (
    FLAT_GAINS,
    CFBGains,
    CommandFilteredBackstepping,
    LQRTracking,
    TrackingModel,
    TwoStepExtraControl,
    position_zeros,
)
from twip.data import IMPLEMENTATION_COLUMNS, load_log, thesis_path
from twip.dynamics import derivative, linear_model
from twip.control import c2d_zoh
from twip.experiments import (
    CONTROL_CASES,
    DEG,
    OBSERVER_CASES,
    control_sim_config,
    design_lqr,
    make_estimators,
    make_truth,
    observer_error,
    rms,
    run_observer,
)
from twip.integrators import rk4
from twip.observers import DMSO, KalmanPredictor, TanhBasis
from twip.params import CORRECTED
from twip.sensors import SensorConfig, SensorSuite

ROOT = Path(__file__).resolve().parents[1] / "docs" / "thesis"
FIG = ROOT / "figures"
GEN = ROOT / "generated"
R2D = 180 / math.pi
SETTLE = 500  # samples excluded as initial transient (5 s)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight"})
STATE = [("Position", "m", 1.0), ("Velocity", "m/s", 1.0), ("Tilt", "deg", R2D), ("Tilt rate", "deg/s", R2D)]
results: dict = {}


def write_table(name: str, rows: list[str]) -> None:
    """Write a table body.  It ends with its own \\bottomrule because \\input is not
    expandable inside a tabular, so a rule placed after the \\input in the .tex fails."""
    (GEN / f"{name}.tex").write_text("\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8")


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------------------
# chapter 4 / 5 / 7 figures


def copy_diagrams():
    from PIL import Image

    src = Path(thesis_path("Pictures", "cropped robot.png"))
    if src.exists():
        im = Image.open(src)
        im.thumbnail((1200, 1600))
        im.convert("RGB").save(FIG / "twip_robot.jpg", quality=88)
        print("  wrote twip_robot.jpg")
    for name, file in (("fbd_wheel", "leftwheel.png"), ("fbd_pendulum", "pendulum fbd.png")):
        p = Path(thesis_path("Thesis", "Diagrams", file))
        if p.exists():
            shutil.copy(p, FIG / f"{name}.png")
            print("  wrote", name)


def uncontrolled(theta0_deg: float, T: float = 10.0):
    """Free response of the corrected plant (v = 0).  Thesis convention: theta = 180 deg upright."""
    x = np.array([0.0, 0.0, (theta0_deg - 180) * DEG, 0.0])
    dt = 0.001
    out = [x]
    for k in range(int(T / dt)):
        x = rk4(lambda t, X: derivative(X, 0.0), x, 0, dt, dt)
        out.append(x)
    X = np.array(out).T
    X[2] = X[2] + math.pi
    return np.arange(X.shape[1]) * dt, X


def fig_uncontrolled():
    for th0, name in ((180.0, "stability_180"), (10.0, "stability_10")):
        t, X = uncontrolled(th0)
        fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.2), sharex=True)
        for ax, (k, (lab, unit, s)) in zip(axs.flat, [(2, STATE[2]), (3, STATE[3]), (0, STATE[0]), (1, STATE[1])]):
            ax.plot(t, X[k] * s)
            ax.set(title=lab, ylabel=unit)
        for ax in axs[1]:
            ax.set_xlabel("Time (s)")
        fig.tight_layout()
        save(fig, name)
        results[name] = {"final_theta_deg": float(X[2, -1] * R2D), "final_x_m": float(X[0, -1]), "first_dx_sign": float(np.sign(X[0, 50]))}


def fig_allan():
    import scipy.io as sio

    path = thesis_path("Tests", "Allan Variance", "allandata.mat")
    if not path.exists():
        return
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    rate = float(m["data"].rate)
    tau = np.logspace(np.log10(2 / rate), 3, 60)
    meas = allan(m["data"].freq, rate, tau)
    # simulated gyro at the same rate from the sensor model
    s = SensorSuite(SensorConfig(bias_init="zero", quantize=False), CORRECTED, np.random.default_rng(0))
    n = len(m["data"].freq)
    sim = np.array([s.measure(np.zeros(4), 0.0, 0.0, 1 / rate).gy for _ in range(n)])
    simr = allan(sim, rate, tau)
    white = allan(np.random.default_rng(1).normal(scale=math.sqrt(0.0012), size=n), rate, tau)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.loglog(meas.tau, meas.osig, label="Measured gyroscope")
    ax.loglog(simr.tau, simr.osig, "--", label="Simulated: white noise + Gauss-Markov bias")
    ax.loglog(white.tau, white.osig, ":", label="Simulated: white noise only")
    ax.set(xlabel=r"Cluster time $\tau$ (s)", ylabel=r"Allan deviation (deg/s)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    save(fig, "allan_variance")


def fig_deadzone_backlash():
    u = np.linspace(-3, 3, 601)
    dp, dm, md, jp, jm = 1.0, 1.0, 1.0, 0.4, 0.4
    D = np.where(u > dp, md * (u - dp) + jp, np.where(u < -dm, md * (u + dm) - jm, 0.0))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for mask in (u < -dm, (u >= -dm) & (u <= dp), u > dp):
        ax.plot(u[mask], D[mask], "k")
    ax.set(xlabel="$u$", ylabel=r"$\mathcal{D}(u)$")
    ax.annotate(r"$\delta_+$", (dp, 0), (dp + 0.1, -0.5))
    ax.annotate(r"$-\delta_-$", (-dm, 0), (-dm - 0.6, 0.3))
    ax.annotate(r"$j_+$", (dp, jp), (dp - 0.5, jp + 0.3))
    save(fig, "deadzone")

    from twip.actuators import Backlash

    t = np.linspace(0, 4 * math.pi, 800)
    uin = 2 * np.sin(t) * np.exp(-0.05 * t)
    bl = Backlash(0.0, 1.0, 0.5, -0.5)
    y = [bl.step(v) for v in uin]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(uin, y, "k")
    ax.plot([-2.5, 2.5], [-3.0, 2.0], ":", color="0.6")
    ax.plot([-2.5, 2.5], [-2.0, 3.0], ":", color="0.6")
    ax.set(xlabel="$u$", ylabel="$y$", xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
    ax.annotate(r"$y=m_b(u-\Delta_b)$", (1.6, 1.1), fontsize=8)
    ax.annotate(r"$y=m_b(u+\Delta_b)$", (-2.4, -0.6), fontsize=8)
    save(fig, "backlash")


# ---------------------------------------------------------------------------
# observer study


EST_ORDER = [
    "KF, nominal Q",
    "KF, inflated Q",
    "Augmented-state KF",
    "DMSO (rev. 7)",
    "DMSO (rev. 8)",
    "DMSO (rev. 8) + sigma-mod",
    "DMSO (rev. 8), Kalman-gain injection",
]
TEX_NAME = {
    "KF, nominal Q": r"KF, nominal $Q_{\mathrm{K}}$",
    "KF, inflated Q": r"KF, inflated $Q_{\mathrm{K}}$",
    "Augmented-state KF": "Augmented-state KF",
    "DMSO (rev. 7)": "DMSO, revision 7",
    "DMSO (rev. 8)": "DMSO",
    "DMSO (rev. 8) + sigma-mod": r"DMSO with $\sigma$-mod",
    "DMSO (rev. 8), Kalman-gain injection": r"DMSO, Kalman-gain $\Km$\,$^\dagger$",
}


def observer_study():
    truths, runs = {}, {}
    for name, case in OBSERVER_CASES.items():
        tr = make_truth(case)
        truths[name] = tr
        runs[name] = {en: run_observer(est, tr) for en, est in make_estimators(tr, case.noise).items()}
    obs = {}
    for name in OBSERVER_CASES:
        tr = truths[name]
        obs[name] = {}
        for en, r in runs[name].items():
            e = rms(observer_error(r, tr), SETTLE)
            fe = rms(r.fhat - tr.f, SETTLE)
            c = r.conditions
            obs[name][en] = {
                "pred_rms": e.tolist(),
                "f_rms": fe.tolist(),
                "conditions": None
                if c is None
                else dict(sigma_max=c.sigma_max_A, gamma=c.gamma, gamma_bound=c.gamma_bound,
                          phi2_design=c.phi_max2_design, phi2_realized=c.phi_max2_realized, satisfied=c.satisfied),
            }
        obs[name]["_truth"] = {"f_rms": rms(tr.f, SETTLE).tolist(), "y_err_rms": rms(tr.y - tr.x, SETTLE).tolist()}
    results["observer"] = obs

    # ultimate bound for O2
    tr = truths["O2"]
    b = TanhBasis()
    Phi = np.array([b(tr.x[:, k]) for k in range(len(tr.u))])
    W, *_ = np.linalg.lstsq(Phi, tr.f.T, rcond=None)
    epsN = float(np.max(np.linalg.norm(tr.f.T - Phi @ W, axis=1)))
    r8 = runs["O2"]["DMSO (rev. 8)"]
    be = r8.conditions.ultimate_bound(epsN)
    en = np.linalg.norm(observer_error(r8, tr), axis=0)
    outside = np.where(en > be)[0]
    inside = np.where(en <= be)[0]
    k_in = int(outside[-1] + 1) if len(outside) else 0
    # adaptation-gain sensitivity (Section 8.2): the revision 8 DMSO at larger fractions
    # of the bound of Theorem 3.1, against the revision 7 observer (gamma = 0.1)
    sens = {}
    for case in ("O2", "O3"):
        sens[case] = {}
        for frac in (0.1, 0.5, 0.9):
            d = DMSO(truths[case].F, truths[case].G, gamma_frac=frac)
            r = run_observer(d, truths[case])
            sens[case][str(frac)] = {
                "gamma": d.gamma,
                "f_rms": rms(r.fhat - truths[case].f, SETTLE).tolist(),
                "pred_rms": rms(observer_error(r, truths[case]), SETTLE).tolist(),
                "satisfied": bool(r.conditions.satisfied),
            }
    results["dmso_gamma_sensitivity"] = sens

    results["O2_bound"] = {
        "eps_N": epsN,
        "b_e": be,
        "first_inside_s": float(inside[0] * 0.01) if len(inside) else None,
        "stays_inside_after_s": float(k_in * 0.01),
        "max_e_after": float(en[k_in:].max()),
        "max_e": float(en.max()),
    }

    t = tr.time[1:]

    # fig O1
    tr1, r1 = truths["O1"], runs["O1"]["DMSO (rev. 8)"]
    fig, axs = plt.subplots(4, 2, figsize=(6.5, 7), sharex=True)
    e1 = observer_error(r1, tr1)
    for i, (lab, unit, s) in enumerate(STATE):
        axs[i, 0].plot(tr1.time, tr1.x[i] * s, label="truth")
        axs[i, 0].plot(tr1.time[1:], r1.xpred[i] * s, "--", label="DMSO")
        axs[i, 0].set(ylabel=f"{lab} ({unit})")
        axs[i, 1].semilogy(tr1.time[1:], np.abs(e1[i]) * s + 1e-16)
        axs[i, 1].set(ylabel="|error|")
    axs[0, 0].legend(fontsize=7)
    axs[0, 0].set_title("State and estimate")
    axs[0, 1].set_title("Estimation error")
    for ax in axs[-1]:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, "O1")

    # fig O2: error norm against b_e
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for en_name, st in (("DMSO (rev. 8)", "-"), ("DMSO (rev. 7)", "--"), ("Augmented-state KF", ":")):
        ax.semilogy(t, np.linalg.norm(observer_error(runs["O2"][en_name], tr), axis=0), st, label=en_name.replace("rev.", "revision"))
    ax.axhline(be, color="k", lw=1)
    ax.annotate(rf"$b_e={be:.3f}$", (t[-1] * 0.8, be * 1.3))
    ax.set(xlabel="Time (s)", ylabel=r"$\|e_k\|$")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "O2_bound")

    # fig O2 uncertainty
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 4.4), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(t, tr.f[i], "k", lw=1.2, label="truth")
        for en_name, st in (("DMSO (rev. 8)", "-"), ("DMSO (rev. 7)", "--"), ("Augmented-state KF", ":")):
            ax.plot(t, runs["O2"][en_name].fhat[i], st, lw=0.9, label=en_name.replace("rev.", "revision"))
        ax.set(ylabel=rf"$f_{i + 1}$")
    axs[0].legend(fontsize=7, ncol=2)
    axs[1].set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, "O2_uncertainty")

    # fig O3: rolling RMS error per state
    tr3 = truths["O3"]
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.8), sharex=True)
    w = 50
    for ax, i in zip(axs.flat, range(4)):
        lab, unit, s = STATE[i]
        for en_name in ("KF, nominal Q", "KF, inflated Q", "Augmented-state KF", "DMSO (rev. 8)", "DMSO (rev. 8), Kalman-gain injection"):
            e = observer_error(runs["O3"][en_name], tr3)[i] * s
            rr = np.sqrt(np.convolve(e**2, np.ones(w) / w, mode="valid"))
            ax.semilogy(tr3.time[w:], rr, lw=0.9, label=en_name.replace("(rev. 8)", "").replace(", Kalman-gain injection", " (Kalman-gain $K_{MSO}$)"))
        ax.set(title=lab, ylabel=f"RMS error ({unit})")
    axs[0, 0].legend(fontsize=6)
    for ax in axs[1]:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, "O3")

    # tables
    def cell(v, scale, fmt="{:.2f}"):
        x = v * scale
        return "$<$0.01" if x < 0.005 else fmt.format(x)

    lines = []
    for en_name in EST_ORDER:
        row = [TEX_NAME[en_name]]
        for case in OBSERVER_CASES:
            e = obs[case][en_name]["pred_rms"]
            row.append(f"{cell(e[1], 1e3)} / {cell(e[2], 1e3)} / {cell(e[3], 1e3)}")
        lines.append(" & ".join(row) + r" \\")
    write_table("tab_obsrms", lines)

    lines = []
    for en_name in EST_ORDER[2:]:
        row = [TEX_NAME[en_name]]
        for case in ("O2", "O3", "O4"):
            f = obs[case][en_name]["f_rms"]
            row.append(f"{cell(f[0], 1e3)} / {cell(f[1], 1e3)}")
        lines.append(" & ".join(row) + r" \\")
    tf = [obs[c]["_truth"]["f_rms"] for c in ("O2", "O3", "O4")]
    lines.append(r"\midrule" + "\n" + " & ".join([r"RMS of true $f$"] + [f"{cell(a[0], 1e3)} / {cell(a[1], 1e3)}" for a in tf]) + r" \\")
    write_table("tab_obsunc", lines)

    lines = []
    for case in OBSERVER_CASES:
        c = obs[case]["DMSO (rev. 8)"]["conditions"]
        ok = "yes" if c["satisfied"] else "no"
        lines.append(
            f"{case} & {c['sigma_max']:.2f} & {c['phi2_design']:.0f} & {c['phi2_realized']:.2f} & "
            f"{c['gamma']:.4f} & {c['gamma'] * c['phi2_realized']:.4f} & {c['gamma_bound']:.3f} & {ok} \\\\"
        )
    write_table("tab_obscond", lines)


# ---------------------------------------------------------------------------
# control study


CONTROLLERS = ("LQR", "Two-step", "CF (flat), no NN", "CF (flat) + NN")
CTRL_TEX = {
    "LQR": "LQR alone",
    "Two-step": "LQR + two-step (rev.~7)",
    "CF (flat), no NN": "Command-filtered, flat output, networks off",
    "CF (flat) + NN": "Command-filtered, flat output, networks on",
}
SEEDS = range(5)


def _make_controller(name, K, m, x0):
    if name == "LQR":
        return LQRTracking(K, m)
    if name == "Two-step":
        return TwoStepExtraControl(K, m)
    return CommandFilteredBackstepping(K, m, x0, gains=FLAT_GAINS, networks=name.endswith("+ NN"), output="flat")


def control_study():
    from dataclasses import replace

    m = TrackingModel.build()
    K = design_lqr()
    ctrl = {}
    runs = {}
    for name, case in CONTROL_CASES.items():
        runs[name] = {}
        for cn in CONTROLLERS:
            stats = []
            for sd in SEEDS:
                cfg = control_sim_config(replace(case, seed=sd))
                c = _make_controller(cn, K, m, np.array(cfg.x0))
                r = simulate(c, KalmanEstimator(sensors=cfg.sensors), cfg)
                xd = np.array([m.x_des(k) for k in range(len(r.time))]).T
                if sd == 0:
                    runs[name][cn] = (r, xd, c)
                if r.fell:
                    stats.append(None)
                    continue
                e = r.x - xd
                last = r.time >= r.time[-1] - 10.0  # slope of the position error over the last 10 s
                drift = np.polyfit(r.time[last], e[0, last], 1)[0]
                stats.append((np.sqrt(np.mean(e[0, SETTLE:] ** 2)), np.sqrt(np.mean(r.v_applied**2)), np.max(np.abs(r.v_applied)), drift))
            ok = [s for s in stats if s is not None]
            ctrl.setdefault(name, {})[cn] = {
                "fell_runs": len(stats) - len(ok),
                "runs": len(stats),
                "rms_pos_err": float(np.mean([s[0] for s in ok])) if ok else None,
                "rms_u": float(np.mean([s[1] for s in ok])) if ok else None,
                "peak_u": float(np.max([s[2] for s in ok])) if ok else None,
                "drift_mps": float(np.mean([abs(s[3]) for s in ok])) if ok else None,
            }
    results["control"] = ctrl
    A, B = linear_model()
    F, G = c2d_zoh(A, B, 0.01)
    results["lqr_slowest_pole"] = float(np.max(np.abs(np.linalg.eigvals(F - G @ K[None, :]))))

    lines = []
    for cn in CONTROLLERS:
        row = [CTRL_TEX[cn]]
        for name in CONTROL_CASES:
            d = ctrl[name][cn]
            cell = "fell" if d["rms_pos_err"] is None else f"{d['rms_pos_err']:.3f} / {d['rms_u']:.2f} / {d['peak_u']:.1f}"
            if d["fell_runs"] and d["rms_pos_err"] is not None:
                cell += f" ({d['fell_runs']} fell)"
            row.append(cell)
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"Command-filtered, as in Section~\ref{sec:cffb} & \multicolumn{4}{c}{unstable for all gains tested (Section~\ref{sec:cfinstab})} \\")
    write_table("tab_ctrleff", lines)

    styles = {"LQR": "-", "Two-step": "--", "CF (flat) + NN": "-"}

    def tracking_fig(name, fname, with_u=False):
        fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.8), sharex=True)
        labels = [("Position error", "m", 0, 1.0), ("Velocity error", "m/s", 1, 1.0), ("Tilt", "deg", 2, R2D), ("Tilt rate", "deg/s", 3, R2D)]
        if with_u:
            labels[3] = ("Applied voltage", "V", None, 1.0)
        for ax, (lab, unit, i, s) in zip(axs.flat, labels):
            for cn, st in styles.items():
                r, xd, _ = runs[name][cn]
                lw = 1.1 if cn.startswith("CF") else 0.8
                label = "Command-filtered (flat)" if cn.startswith("CF") else cn
                if i is None:
                    ax.plot(r.time[:-1], r.v_applied, st, lw=lw, label=label)
                else:
                    yv = (r.x[i] - xd[i]) if i < 2 else r.x[i]
                    ax.plot(r.time, yv * s, st, lw=lw, label=label)
            ax.set(title=lab, ylabel=unit)
        axs[0, 0].legend(fontsize=7)
        for ax in axs[1]:
            ax.set_xlabel("Time (s)")
        fig.tight_layout()
        save(fig, fname)

    tracking_fig("C2", "C2")
    tracking_fig("C4", "C4", with_u=True)

    # Figure 8.5: (a) eigenvalues against bandwidth, both designs; (b) filter errors of the corrected design
    A, B = linear_model()
    zrhp = float(np.max(position_zeros(A, B).real))
    scales = np.logspace(-0.7, 1.0, 25)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.4))
    worst = []
    for lab, c in {"as written, $c=(1,2,6,12)$": (1, 2, 6, 12), "as written, $c=(2,4,12,25)$": (2, 4, 12, 25)}.items():
        mx = [np.linalg.eigvals(_cfb_jacobian(CFBGains(c=c, wn=tuple(s * w for w in (20, 40, 60))))).real.max() for s in scales]
        worst.append(min(mx))
        ax.semilogx(scales * 40, mx, marker=".", ms=3, label=lab)
    mx = [np.linalg.eigvals(_cfb_jacobian(CFBGains(c=FLAT_GAINS.c, wn=tuple(s * w for w in (20, 40, 60))), flat=True)).real.max() for s in scales]
    best_flat = mx
    ax.semilogx(scales * 40, mx, "k", marker=".", ms=3, label="flat output (corrected)")
    ax.axhline(zrhp, color="k", ls="--", lw=0.8)
    ax.annotate(f"RHP zero +{zrhp:.2f}", (scales[0] * 40, zrhp + 1.0), fontsize=7)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set(xlabel=r"$\varpi_2$ (rad/s)", ylabel=r"max Re$\lambda$ (rad/s)", title="(a) closed-loop eigenvalues")
    ax.legend(fontsize=6)

    chi_rms, wn2 = [], []
    for s in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        g = replace(FLAT_GAINS, wn=tuple(s * w for w in FLAT_GAINS.wn))
        cfg = control_sim_config(CONTROL_CASES["C2"])
        c = CommandFilteredBackstepping(K, m, np.array(cfg.x0), gains=g, networks=False, output="flat")
        r = simulate(c, KalmanEstimator(sensors=cfg.sensors), cfg)
        wn2.append(g.wn[1])
        chi_rms.append(None if r.fell else np.sqrt(np.mean(np.array(c.log["chi"])[100:] ** 2, axis=0)))
    ok = [i for i, v in enumerate(chi_rms) if v is not None]
    for j, lab in enumerate((r"$\chi_1$", r"$\chi_2$", r"$\chi_3$")):
        bx.loglog([wn2[i] for i in ok], [chi_rms[i][j] for i in ok], marker="o", ms=3, label=lab)
    for i, v in enumerate(chi_rms):
        if v is None:
            bx.axvline(wn2[i], color="r", lw=0.8, ls=":")
    # Appendix A: filter bandwidth at most 1/5 of the sample rate (2 pi / 0.01 s), applied to the
    # fastest filter varpi_3 = 1.5 varpi_2
    wn2_limit = 0.2 * (2 * math.pi / 0.01) / 1.5
    bx.axvline(wn2_limit, color="0.4", lw=0.8, ls="--")
    bx.annotate("App. A limit", (wn2_limit * 1.03, bx.get_ylim()[0] * 1.5), fontsize=7, rotation=90)
    bx.set(xlabel=r"$\varpi_2$ (rad/s), $\varpi_1:\varpi_2:\varpi_3=1:2:3$", ylabel="RMS filter error", title="(b) corrected design, case C2")
    ticks = [20, 30, 40, 60, 80, 120]
    bx.set_xticks(ticks)
    bx.set_xticklabels([str(t) for t in ticks])
    bx.minorticks_off()
    bx.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "cfb_bandwidth")
    results["cfb"] = {
        "rhp_zero": zrhp,
        "min_max_real_eig_as_written": float(min(worst)),
        "flat_max_real_eig_at_design_bandwidth": float(best_flat[int(np.argmin(np.abs(scales - 1.0)))]),
        "flat_chi_rms": {str(w): (None if v is None else v.tolist()) for w, v in zip(wn2, chi_rms)},
        "flat_weight_w": float(CommandFilteredBackstepping(K, m, np.zeros(4), output="flat").w),
        "flat_a2": float(CommandFilteredBackstepping(K, m, np.zeros(4), output="flat").a2),
        "flat_b": float(CommandFilteredBackstepping(K, m, np.zeros(4), output="flat").b),
        "A2": float(m.A2),
    }
    gain_sweeps(K, m)


# Gain grids behind the sweep counts quoted in Sections 6.7 and 8.3.1.  Every point
# of the linear grid satisfies the gain conditions of Theorem 6.1 (c2 > 1, others > 1/2).
LIN_GRID_C = list(itertools.product((0.6, 1.0, 2.0), (1.2, 2.0, 4.0), (1.0, 3.0, 6.0, 10.0), (1.0, 4.0, 12.0, 25.0)))
LIN_GRID_SCALE = (0.5, 1.0, 2.0, 4.0, 8.0)  # applied to varpi = (20, 40, 60) rad/s
SIM_GRID_C = list(itertools.product((0.4, 0.8, 1.5, 3.0), (1.2, 1.5, 3.0, 6.0), (5.0, 10.0), (6.0, 12.0)))


def gain_sweeps(K, m):
    """Linearized closed-loop stability over LIN_GRID (720 points) for the design as
    written and the flat design (w = a2^2 and w = 1), and nonlinear closed-loop runs of
    the flat design (networks off) over SIM_GRID in cases C2 and C4."""
    lin = {"as_written": [], "flat": [], "flat_w1": []}
    for c in LIN_GRID_C:
        for s in LIN_GRID_SCALE:
            g = CFBGains(c=c, wn=tuple(s * w for w in (20.0, 40.0, 60.0)))
            lin["as_written"].append(np.linalg.eigvals(_cfb_jacobian(g)).real.max())
            lin["flat"].append(np.linalg.eigvals(_cfb_jacobian(g, flat=True)).real.max())
            lin["flat_w1"].append(np.linalg.eigvals(_cfb_jacobian(g, flat=True, weight=1.0)).real.max())
    out = {
        k: {"points": len(v), "stable": int(np.sum(np.array(v) < 0)), "min_max_real": float(np.min(v)), "max_max_real": float(np.max(v))}
        for k, v in lin.items()
    }
    sim = []
    for c in SIM_GRID_C:
        g = CFBGains(c=c, wn=FLAT_GAINS.wn, zeta=FLAT_GAINS.zeta)
        row = {"c": c}
        for case in ("C2", "C4"):
            cfg = control_sim_config(CONTROL_CASES[case])
            ctl = CommandFilteredBackstepping(K, m, np.array(cfg.x0), gains=g, networks=False, output="flat")
            r = simulate(ctl, KalmanEstimator(sensors=cfg.sensors), cfg)
            xd = np.array([m.x_des(k) for k in range(len(r.time))]).T
            row[case] = None if r.fell else float(np.sqrt(np.mean((r.x[0, SETTLE:] - xd[0, SETTLE:]) ** 2)))
        sim.append(row)
    both = [r for r in sim if r["C2"] is not None and r["C4"] is not None]
    out["closed_loop"] = {
        "gain_sets": len(sim),
        "balanced_both": len(both),
        "fell": [r["c"] for r in sim if r["C2"] is None or r["C4"] is None],
        "runs": sim,
    }
    # w = 1 at the design gains, closed loop, case C2
    cfg = control_sim_config(CONTROL_CASES["C2"])
    ctl = CommandFilteredBackstepping(K, m, np.array(cfg.x0), gains=FLAT_GAINS, networks=False, output="flat", weight=1.0)
    r = simulate(ctl, KalmanEstimator(sensors=cfg.sensors), cfg)
    xd = np.array([m.x_des(k) for k in range(len(r.time))]).T
    out["flat_w1_C2"] = None if r.fell else float(np.sqrt(np.mean((r.x[0, SETTLE:] - xd[0, SETTLE:]) ** 2)))
    results["cfb_sweeps"] = out


def perturbation_check():
    """The revision-7 torque-constant error (km x2.5) on the corrected plant: whether each
    controller balances (case C2, five seeds), and how much of the resulting uncertainty
    the observer basis can represent (case O2)."""
    from dataclasses import replace

    from twip.experiments import PERTURBATION, ObserverCase

    p25 = CORRECTED.scaled(**{**PERTURBATION, "km": 2.5})
    m = TrackingModel.build()
    K = design_lqr()
    out = {"controllers_C2_km2.5": {}}
    for cn in CONTROLLERS:
        fell = 0
        for sd in SEEDS:
            cfg = control_sim_config(replace(CONTROL_CASES["C2"], plant=p25, seed=sd))
            r = simulate(_make_controller(cn, K, m, np.array(cfg.x0)), KalmanEstimator(sensors=cfg.sensors), cfg)
            fell += int(r.fell)
        out["controllers_C2_km2.5"][cn] = {"runs": len(SEEDS), "fell": fell}
    b = TanhBasis()
    for label, plant in (("km1.25", CORRECTED.scaled(**PERTURBATION)), ("km2.5", p25)):
        tr = make_truth(ObserverCase("O2", noise=False, plant=plant))
        Phi = np.array([b(tr.x[:, k]) for k in range(len(tr.u))])
        W, *_ = np.linalg.lstsq(Phi, tr.f.T, rcond=None)
        res = tr.f.T - Phi @ W
        out[f"O2_{label}"] = {
            "eps_N": float(np.max(np.linalg.norm(res, axis=1))),
            "f_rms": rms(tr.f, SETTLE).tolist(),
            "fit_residual_rms": rms(res.T, SETTLE).tolist(),
        }
    results["perturbation_check"] = out


def _cfb_jacobian(g: CFBGains, flat: bool = False, weight: float | None = None) -> np.ndarray:
    """Continuous-time closed loop (plant + command filters, networks off, nominal linear model).

    ``weight`` is the attitude weight ``w`` of the flat design (default ``a2**2``)."""
    m = TrackingModel.build()
    A, B = m.A, m.B
    b = m.B1 / m.B2
    a2 = m.A2 - b * m.A4
    W = a2**2 if weight is None else weight

    def f(s):
        x = s[:4]
        q = s[4:].reshape(3, 2)
        if flat:
            z = np.array([x[0] - b * x[2], x[1] - b * x[3] - q[0, 0], x[2] - q[1, 0], x[3] - q[2, 0]])
            F1 = (m.A1 - b * m.A3) * x[1] + (A[1, 3] - b * A[3, 3]) * x[3] - q[0, 1]
        else:
            z = np.array([x[0], x[1] - q[0, 0], x[2] - q[1, 0], x[3] - q[2, 0]])
            F1 = m.A1 * x[1] + A[1, 3] * x[3] - q[0, 1]
        F2 = m.A3 * x[1] + m.A4 * x[2] + A[3, 3] * x[3] - q[2, 1]
        u = (-F2 - g.c[3] * z[3] - z[2]) / m.B2
        if flat:
            a = (-g.c[0] * z[0], (-F1 - g.c[1] * z[1] - z[0]) / a2, q[1, 1] - g.c[2] * z[2] - (a2 / W) * z[1])
        else:
            a = (-g.c[0] * z[0], (-F1 - m.B1 * u - g.c[1] * z[1] - z[0]) / m.A2, q[1, 1] - g.c[2] * z[2] - m.A2 * z[1])
        dq = []
        for i in range(3):
            w, zt = g.wn[i], g.zeta[i]
            dq += [q[i, 1], -2 * zt * w * q[i, 1] - w**2 * (q[i, 0] - a[i])]
        return np.concatenate([A @ x + B * u, dq])

    h = 1e-6
    return np.column_stack([(f(h * e) - f(-h * e)) / (2 * h) for e in np.eye(10)])


# ---------------------------------------------------------------------------
# hardware replay


def hardware():
    path = thesis_path("Thesis", "Thesis Programs", "Implementation", "implementationtest5.txt")
    if not path.exists():
        return
    L = load_log(path, IMPLEMENTATION_COLUMNS)
    r = CORRECTED.r
    th, w = L["pitchc"] * DEG, L["gy"] * DEG
    acc = np.arctan2(-L["ax"], L["az"])
    u = np.trunc(L["control"]) / 5904.5
    t = np.concatenate(([0.0], np.cumsum(L["dt"][:-1])))
    dt = float(np.median(L["dt"]))
    A, B = linear_model()
    F, G = c2d_zoh(A, B, dt)
    s = SensorConfig()
    q = s.meters_per_count
    Gw = np.array([[dt**2 / 2, 0], [dt, 0], [0, dt**2 / 2], [0, dt]])
    Q = Gw @ np.diag([1.0, 5.0**2]) @ Gw.T + 1e-10 * np.eye(4)

    # The encoders count wheel rotation relative to the body, x / r + theta (twip.sensors).
    # The firmware's tilt has the sign of theta, but its position is mirrored:
    # pos = -(x + r theta) (scripts/validate_plant.py, frame_evidence and the input-free
    # tests).  So x = -pos - r * tilt in the model frame the estimators use.
    pos_sign = -1.0

    def to_model(pos, vel, tilt, rate):
        return pos_sign * pos - r * tilt, pos_sign * vel - r * rate

    def run(est, tilt):
        y = np.vstack([*to_model(L["x"], L["xdot"], tilt, w), tilt, w])
        est.initialize(y[:, 0])
        xp = np.zeros_like(y)
        fh = np.full((2, y.shape[1]), np.nan)
        xp[:, 0] = y[:, 0]
        for k in range(y.shape[1] - 1):
            xp[:, k + 1] = est.step(y[:, k], u[k])
            f = getattr(est, "fhat", None)
            if f is not None and len(f):
                fh[:, k] = f
        return xp, fh

    R_comp = np.diag([q**2 / 12, 2 * q**2 / 12 / dt**2, (0.5 * DEG) ** 2, (s.gyro_white_std * DEG) ** 2])
    R_raw = R_comp.copy()
    R_raw[2, 2] = (2 * DEG) ** 2
    dm_c, dm_fh = run(DMSO(F, G), th)
    kf_c, _ = run(KalmanPredictor(F, G, Q, R_comp), th)
    dm_raw, _ = run(DMSO(F, G), acc)
    kf_raw, _ = run(KalmanPredictor(F, G, Q, R_raw), acc)
    s0 = 310  # balancing starts (as in lqrkalmantest1v2.m)
    tt = t[s0:] - t[s0]

    fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.8), sharex=True)
    meas = [*to_model(L["x"], L["xdot"], th, w), th, w]
    onboard = [*to_model(L["xhat"], L["xhatdot"], L["pitchm"] * DEG, L["gyhat"] * DEG), L["pitchm"] * DEG, L["gyhat"] * DEG]
    for ax, i in zip(axs.flat, range(4)):
        lab, unit, sc = STATE[i]
        ax.plot(tt, meas[i][s0:] * sc, color="0.6", lw=0.8, label="measured")
        ax.plot(tt, onboard[i][s0:] * sc, lw=0.8, label="on-board DMSO (v9)")
        ax.plot(tt, dm_c[i, s0:] * sc, "--", lw=0.8, label="DMSO (rev. 8)")
        ax.set(title=lab, ylabel=unit)
    axs[0, 0].legend(fontsize=6)
    for ax in axs[1]:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, "hw_states")

    # complementary vs estimators fed raw accelerometer tilt; score by consistency with the gyro
    def gyro_consistency(tilt):
        d = np.diff(tilt[s0:]) / L["dt"][s0:-1]
        return float(np.sqrt(np.mean((d - w[s0:-1]) ** 2)))

    cons = {
        "Accelerometer tilt": gyro_consistency(acc),
        "Complementary filter": gyro_consistency(th),
        "KF on accelerometer tilt": gyro_consistency(kf_raw[2]),
        "DMSO on accelerometer tilt": gyro_consistency(dm_raw[2]),
        "KF on complementary tilt": gyro_consistency(kf_c[2]),
        "DMSO on complementary tilt": gyro_consistency(dm_c[2]),
    }
    results["hardware_gyro_consistency_dps"] = {k: v * R2D for k, v in cons.items()}
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(tt, acc[s0:] * R2D, color="0.75", lw=0.7, label="accelerometer tilt")
    ax.plot(tt, th[s0:] * R2D, lw=1.0, label="complementary filter")
    ax.plot(tt, kf_raw[2, s0:] * R2D, "--", lw=0.8, label="KF on accelerometer tilt")
    ax.plot(tt, dm_raw[2, s0:] * R2D, ":", lw=0.9, label="DMSO on accelerometer tilt")
    ax.set(xlabel="Time (s)", ylabel="Tilt (deg)")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "hw_complementary")

    fig, axs = plt.subplots(2, 1, figsize=(6.5, 4.0), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(tt, dm_fh[i, s0:], lw=0.8)
        ax.set(ylabel=rf"$\hat f_{i + 1}$")
    axs[1].set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, "hw_uncertainty")
    results["hardware_fhat_rms"] = rms(np.nan_to_num(dm_fh[:, s0:]), 0).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["ch4", "ch5", "ch7", "observer", "control", "perturbation", "hardware"])
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    rp = GEN / "results.json"
    if rp.exists():
        results.update(json.loads(rp.read_text()))
    steps = {
        "ch4": copy_diagrams,
        "ch5": fig_uncontrolled,
        "ch7": lambda: (fig_allan(), fig_deadzone_backlash()),
        "observer": observer_study,
        "control": control_study,
        "perturbation": perturbation_check,
        "hardware": hardware,
    }
    for name, fn in steps.items():
        if args.only in (None, name):
            print(name)
            fn()
    rp.write_text(json.dumps(results, indent=1, default=float))
    print("wrote", rp)


if __name__ == "__main__":
    main()
