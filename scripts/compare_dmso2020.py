"""Compare the DMSO of the 2020 ACC paper with the revision-8 DMSO.

    uv run python scripts/compare_dmso2020.py               # everything
    uv run python scripts/compare_dmso2020.py --only paper  # paper, matrix, gamma or hardware

The paper is Stumfoll, Yao and Balakrishnan, "Neural Network Based Discrete Time
Modified State Observer: Stability Analysis and Case Study", ACC 2020.  Its observer
is :class:`twip.observers.DMSO2020`.  Four steps:

* ``paper``: the paper's own case study (Section IV) rebuilt from its Table 1, on its
  linear model (eq. 41, which is :func:`twip.legacy.dynamics.linear_model`), at the
  stated 10 ms and at 1 ms, which is the step its figures match.
* ``matrix``: the observer cases O1-O4 of thesis Chapter 8 (corrected plant), with the
  paper's observer next to the revision-8 DMSO and the augmented-state Kalman filter.
* ``gamma``: the paper's leakage condition (10), ``0 < Gamma < 2``, near its upper end.
* ``hardware``: the v9 balancing logs ``implementationtest4/5.txt`` replayed through both
  observers.  Needs the thesis data folder (``TWIP_THESIS_ROOT``); skipped without it.

Writes ``docs/dmso2020/results.json`` and PNG figures next to it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twip.control import c2d_zoh, dlqr
from twip.data import IMPLEMENTATION_COLUMNS, load_log, thesis_path
from twip.dynamics import linear_model
from twip.experiments import OBSERVER_CASES, make_estimators, make_truth, observer_error, rms, run_observer
from twip.legacy.dynamics import linear_model as linear_model_rev7
from twip.observers import B_MSO, DMSO, DMSO2020, TanhBasis
from twip.params import CORRECTED, THESIS
from twip.sensors import DEG

OUT = Path(__file__).resolve().parents[1] / "docs" / "dmso2020"
SETTLE = 500  # samples excluded as initial transient, as in thesis Chapter 8
results: dict = {}

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------------------
# the paper's case study


PAPER_Q = np.diag([10, 10, 1e-5, 1e-4])  # Section IV: R = 1000, Q = diag[10, 10, 1e-5, 1e-4]
PAPER_R = 1000.0
PAPER_X0 = np.array([1.0, 0.3, 10.0, 1.0])  # eq. (50)
PAPER_STEPS = 10_000  # the figures span 100 "s" at 10 ms


def paper_truth(dt: float, noise: dict | None = None, seed: int = 0) -> dict:
    """Case I (and with ``noise`` Case II) of the paper.

    Nominal model: Table 1 "DMSO values" = :data:`THESIS`.  True model: Table 1 "True
    values" = ``THESIS.with_thesis_uncertainty(l_scale=1.0)``.  Both on eq. (41).  The
    LQR is designed for the true system (45), as the text says, and acts on the true
    state (46).  ``f`` is eq. (48), rows 2 and 4.
    """
    Ae, Be = linear_model_rev7(THESIS)
    At, Bt = linear_model_rev7(THESIS.with_thesis_uncertainty(l_scale=1.0))
    Fe, Ge = c2d_zoh(Ae, Be, dt)
    Ft, Gt = c2d_zoh(At, Bt, dt)
    Ge, Gt = Ge.ravel(), Gt.ravel()
    K = dlqr(Ft, Gt.reshape(4, 1), PAPER_Q, PAPER_R).ravel()
    N = PAPER_STEPS
    X = np.zeros((4, N + 1))
    U = np.zeros(N)
    X[:, 0] = PAPER_X0
    for k in range(N):
        U[k] = -K @ X[:, k]
        X[:, k + 1] = Ft @ X[:, k] + Gt * U[k]
    Y = X.copy()
    if noise:  # white noise on the tilt and gyro channels (Table 3 sigma_r^2 read as variances)
        rng = np.random.default_rng(seed)
        Y[2] += rng.normal(0, math.sqrt(noise["tilt_var"]), N + 1)
        Y[3] += rng.normal(0, math.sqrt(noise["gyro_var"]), N + 1)
    f = B_MSO.T @ (X[:, 1:] - Fe @ X[:, :-1] - np.outer(Ge, U))
    return dict(X=X, Y=Y, U=U, f=f, Fe=Fe, Ge=Ge, K=K, rank_A=int(np.linalg.matrix_rank(Ae)))


def run_on(est, X: np.ndarray, Y: np.ndarray, U: np.ndarray):
    N = len(U)
    xp = np.zeros((4, N))
    fh = np.zeros((2, N))
    est.initialize(Y[:, 0])
    for k in range(N):
        xp[:, k] = est.step(Y[:, k], U[k])
        fh[:, k] = B_MSO.T @ (est.Bm @ est.fhat)
    return xp, fh


def paper_estimators(Fe, Ge) -> dict:
    return {
        "K = I, no network (alpha = 0)": DMSO2020(Fe, Ge, alpha=0.0, gamma=0.0),
        "DMSO2020, Gamma = 0": DMSO2020(Fe, Ge, gamma=0.0),
        "DMSO2020, Gamma = 1e-4": DMSO2020(Fe, Ge, gamma=1e-4),
        "DMSO2020, Gamma = 1e-3 (default)": DMSO2020(Fe, Ge),
        "DMSO2020, Gamma = 1e-2": DMSO2020(Fe, Ge, gamma=1e-2),
        "DMSO2020, Gamma = 1e-1": DMSO2020(Fe, Ge, gamma=1e-1),
        "DMSO2020, Gamma = 1e-3, Frobenius norm": DMSO2020(Fe, Ge, leak_norm="fro"),
        "DMSO rev. 8 (A = 0.5 I)": DMSO(Fe, Ge),
    }


def paper():
    out = {}
    traces = {}
    for dt in (0.01, 0.001):
        tr = paper_truth(dt)
        X, U, f = tr["X"], tr["U"], tr["f"]
        f_rms = rms(f, SETTLE)
        # features of Figures 1 and 2, located by step index (the figures' "Time (sec)" is step * 0.01)
        k_xmin, k_vmin = int(np.argmin(X[0])), int(np.argmin(X[1]))
        k_fpk = SETTLE // 10 + int(np.argmax(f[0, SETTLE // 10 :]))
        feat = {
            "f_at_k0": f[:, 0].tolist(),
            "position_min": [float(X[0, k_xmin]), k_xmin],
            "velocity_min": [float(X[1, k_vmin]), k_vmin],
            "tilt_min": [float(X[2].min()), int(np.argmin(X[2]))],
            "tilt_rate_min": float(X[3].min()),
            "f1_peak": [float(f[0, k_fpk]), k_fpk],
            "u_max_V": float(np.abs(U).max()),
            "state_range": np.ptp(X, axis=1).tolist(),
        }
        ests = {}
        for name, est in paper_estimators(tr["Fe"], tr["Ge"]).items():
            xp, fh = run_on(est, X, X, U)
            e = X[:, 1:] - xp
            c = est.conditions()
            ests[name] = {
                "pred_rms_after": rms(e, SETTLE).tolist(),
                "max_abs_err": np.abs(e).max(axis=1).tolist(),
                "f_err_over_f_rms": (rms(fh - f, SETTLE) / f_rms).tolist(),
                "f_err_rms_first": rms((fh - f)[:, :SETTLE], 0).tolist(),
                "sigma_max_A": c.sigma_max_A,
            }
            if isinstance(est, DMSO2020):
                ests[name].update(paper_conditions=list(c.paper_conditions), weights_contractive=c.weights_contractive)
            if name.endswith("(default)"):
                traces[dt] = (f, fh)
        out[f"dt={dt}"] = {"features": feat, "f_rms_after": f_rms.tolist(), "K_lqr": tr["K"].tolist(), "estimators": ests}
    out["rank_A_eq41"] = paper_truth(0.01)["rank_A"]

    # Case II: noise.  Estimate error against the measurement's own error, tilt and tilt rate.
    noise = {"tilt_var": 4.3097e-6, "gyro_var": 0.0012}
    tr = paper_truth(0.001, noise=noise)
    X, Y, U = tr["X"], tr["Y"], tr["U"]
    case2 = {"noise": noise}
    for name, est in (
        ("DMSO2020 (K = I)", DMSO2020(tr["Fe"], tr["Ge"])),
        ("DMSO rev. 8 (A = 0.5 I)", DMSO(tr["Fe"], tr["Ge"])),
    ):
        xp, _ = run_on(est, X, Y, U)
        e = rms(X[:, 1:] - xp, SETTLE)
        m = rms(Y[:, 1:] - X[:, 1:], SETTLE)
        case2[name] = {"tilt_err_over_meas": float(e[2] / m[2]), "rate_err_over_meas": float(e[3] / m[3])}
    out["case2"] = case2
    results["paper"] = out

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    for j, dt in enumerate((0.001, 0.01)):
        f, fh = traces[dt]
        t = np.arange(f.shape[1]) * 0.01
        for i in range(2):
            ax = axs[i, j]
            ax.plot(t, f[i], "r", lw=1.2, label="true (eq. 48)")
            ax.plot(t, fh[i], "b", lw=0.8, label="DMSO2020 estimate")
            ax.set_ylim(np.array([(-0.5, 0.1), (-0.3, 0.1)][i]) * (1 if dt == 0.001 else 10))  # the paper's axes at 1 ms
        axs[0, j].set_title(rf"$\Delta t$ = {dt * 1000:g} ms")
    axs[0, 0].set_ylabel("Acceleration uncertainty")
    axs[1, 0].set_ylabel("Tilt acceleration uncertainty")
    for ax in axs[1]:
        ax.set_xlabel("step x 0.01 (the paper's 'Time (sec)')")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "paper_fig2")


# ---------------------------------------------------------------------------
# thesis Chapter 8 observer cases


MATRIX_ROWS = (
    "Augmented-state KF",
    "DMSO (rev. 7)",
    "DMSO (rev. 8)",
    "DMSO (rev. 8) + sigma-mod",
    "DMSO (ACC 2020)",
    "DMSO (ACC 2020), Gamma = 0",
    "DMSO (ACC 2020), rev. 8 injection",
    "DMSO (ACC 2020), rev. 8 injection and basis",
)


def matrix():
    out = {}
    o2 = None
    for name, case in OBSERVER_CASES.items():
        tr = make_truth(case)
        ests = {k: v for k, v in make_estimators(tr, case.noise, acc2020=True).items() if k in MATRIX_ROWS}
        # which choice costs what: no leakage; A = 0.5 I as in revision 8; and A = 0.5 I with the
        # revision-8 basis, which with the 2-norm is exactly the revision-8 sigma-mod row
        ests["DMSO (ACC 2020), Gamma = 0"] = DMSO2020(tr.F, tr.G, gamma=0.0)
        ests["DMSO (ACC 2020), rev. 8 injection"] = DMSO2020(tr.F, tr.G, Km=tr.F - 0.5 * np.eye(4))
        ests["DMSO (ACC 2020), rev. 8 injection and basis"] = DMSO2020(
            tr.F, tr.G, alpha=0.01, gamma=0.001, Km=tr.F - 0.5 * np.eye(4), basis=TanhBasis()
        )
        runs = {k: run_observer(v, tr) for k, v in ests.items()}
        out[name] = {
            k: {
                "pred_rms": rms(observer_error(r, tr), SETTLE).tolist(),
                "f_rms": rms(r.fhat - tr.f, SETTLE).tolist(),
            }
            for k, r in runs.items()
        }
        out[name]["_truth"] = {"f_rms": rms(tr.f, SETTLE).tolist(), "y_err_rms": rms(tr.y - tr.x, SETTLE).tolist()}
        c = runs["DMSO (ACC 2020)"].conditions
        out[name]["_acc2020_conditions"] = {
            "sigma_max_A": c.sigma_max_A,
            "phi_max2_realized": c.phi_max2_realized,
            "paper_conditions": list(c.paper_conditions),
            "weights_contractive": c.weights_contractive,
        }
        # the ACC 2020 law with the 2-norm is the revision-8 sigma-modification
        out[name]["_equivalence_max_abs_diff"] = float(
            np.max(np.abs(runs["DMSO (ACC 2020), rev. 8 injection and basis"].xpred - runs["DMSO (rev. 8) + sigma-mod"].xpred))
        )
        if name == "O2":
            o2 = (tr, runs)
    results["matrix"] = out

    tr, runs = o2
    t = tr.time[1:]
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(t, tr.f[i], "k", lw=1.2, label="truth")
        for k, st in (("DMSO (rev. 8)", "-"), ("DMSO (ACC 2020)", "--"), ("Augmented-state KF", ":")):
            ax.plot(t, runs[k].fhat[i], st, lw=0.9, label=k)
        ax.set(ylabel=rf"$f_{i + 1}$")
    axs[0].legend(fontsize=7, ncol=2)
    axs[1].set_xlabel("Time (s)")
    fig.tight_layout()
    save(fig, "O2_uncertainty")


# ---------------------------------------------------------------------------
# condition (10)


def first_difference_check(draws: int = 2000, seed: int = 1) -> float:
    """Largest relative error of the closed form of ``Delta V_k`` for the paper's law.

    With the 2-norm (``||I - alpha phi phi^T|| = 1``), ``s = A e + eps``, ``eb = W~^T phi``
    and ``a = alpha ||phi||^2``, the difference of ``V = e^T e + tr(W~^T W~) / alpha``
    along (16) and (6) is

        (1 + a)|s|^2 + 2 (a + Gamma) s^T eb - (1 - a - 2 Gamma)|eb|^2 - |e|^2
        - 2 Gamma (s + eb)^T W^T phi - Gamma (2 - Gamma) / alpha |W~|^2
        + 2 Gamma (1 - Gamma) / alpha tr(W~^T W) + Gamma^2 / alpha |W|^2.

    The paper's (32) has ``-(1 - a)|eb|^2``: the ``+2 Gamma |eb|^2`` is lost between
    (30) and (31).  This compares the closed form with the difference computed
    directly from the two update equations on random draws.
    """
    rng = np.random.default_rng(seed)
    n, l = 4, 5
    worst = 0.0
    for _ in range(draws):
        alpha, Gam = 10 ** rng.uniform(-3, -1), rng.uniform(0, 2)
        phi = np.r_[1.0, np.tanh(rng.normal(0, 2, n))]
        a = alpha * phi @ phi
        A, e, eps = rng.normal(0, 0.3, (n, n)), rng.normal(0, 1, n), rng.normal(0, 0.1, n)
        W, Wt = rng.normal(0, 1, (l, n)), rng.normal(0, 1, (l, n))
        e1 = A @ e + Wt.T @ phi + eps  # (16)
        What1 = (1 - Gam) * (W - Wt) + alpha * np.outer(phi, e1)  # (6), ||M|| = 1
        Wt1 = W - What1
        dV = e1 @ e1 - e @ e + (np.sum(Wt1**2) - np.sum(Wt**2)) / alpha
        s, eb = A @ e + eps, Wt.T @ phi
        closed = (
            (1 + a) * s @ s + 2 * (a + Gam) * s @ eb - (1 - a - 2 * Gam) * eb @ eb - e @ e
            - 2 * Gam * (s + eb) @ (W.T @ phi) - Gam * (2 - Gam) / alpha * np.sum(Wt**2)
            + 2 * Gam * (1 - Gam) / alpha * np.sum(Wt * W) + Gam**2 / alpha * np.sum(W**2)
        )
        worst = max(worst, abs(dV - closed) / max(1.0, abs(dV)))
    return float(worst)


def gamma():
    """Leakage near the top of 0 < Gamma < 2 on the paper's own case (1 ms)."""
    tr = paper_truth(0.001)
    X, U = tr["X"], tr["U"]
    out = {"first_difference_closed_form_max_rel_err": first_difference_check()}
    for g in (0.5, 1.0, 1.9, 1.98, 1.99):
        for norm in ("2", "fro"):
            est = DMSO2020(tr["Fe"], tr["Ge"], gamma=g, leak_norm=norm)
            est.initialize(X[:, 0])
            diverged = None
            with np.errstate(all="ignore"):
                for k in range(len(U)):
                    est.step(X[:, k], U[k])
                    if not np.all(np.isfinite(est.W)) or np.abs(est.W).max() > 1e6:
                        diverged = k
                        break
            c = est.conditions()
            out[f"Gamma={g}, {norm}-norm"] = {
                "paper_conditions": list(c.paper_conditions),
                "weights_contractive": c.weights_contractive,
                "diverged_at_step": diverged,
                "max_abs_W": float(np.abs(est.W).max()),
            }
    results["gamma"] = out


# ---------------------------------------------------------------------------
# hardware log


def hardware():
    """Replay the two v9 balancing logs (``implementationtest4/5.txt``) through both observers.

    There is no truth on hardware, so each observer is scored by its one-step
    prediction residual ``y_{k+1} - xhat_{k+1|k}`` and by the consistency of its tilt
    with the gyro, ``rms(d tilt/dt - gyro)``.  The frame is the one the thesis
    replays use (``scripts/thesis_rev8.py``, ``hardware``): the logged position is
    taken as mirrored, ``x = -pos - r * tilt``.  Scoring starts four samples after
    the motor command first goes nonzero (sample 310 of test 5, as in the thesis).
    """
    out = {}
    for n_test in (4, 5):
        path = thesis_path("Thesis", "Thesis Programs", "Implementation", f"implementationtest{n_test}.txt")
        if not path.exists():
            print("  skipped: thesis data folder not found (set TWIP_THESIS_ROOT)")
            return
        out[f"test{n_test}"] = _replay(load_log(path, IMPLEMENTATION_COLUMNS))
    results["hardware"] = out


def _replay(L: dict) -> dict:
    r = CORRECTED.r
    th, w = L["pitchc"] * DEG, L["gy"] * DEG
    u = np.trunc(L["control"]) / 5904.5
    dt = float(np.median(L["dt"]))
    F, G = c2d_zoh(*linear_model(), dt)
    G = G.ravel()
    y = np.vstack([-L["x"] - r * th, -L["xdot"] - r * w, th, w])
    s0 = int(np.flatnonzero(L["control"])[0]) + 4  # balancing starts

    def gyro_consistency(tilt: np.ndarray) -> float:  # as in thesis_rev8.hardware, deg/s
        d = np.diff(tilt[s0:]) / L["dt"][s0:-1]
        return float(np.sqrt(np.mean((d - w[s0:-1]) ** 2)) / DEG)

    def score(xt: np.ndarray, fh: np.ndarray | None) -> dict:
        """``xt[:, k]`` is the prediction of ``y[:, k]`` made at step ``k - 1``."""
        out = {"residual_rms": rms(y[:, s0:] - xt[:, s0:], 0).tolist(), "gyro_consistency_dps": gyro_consistency(xt[2])}
        if fh is not None:
            out["fhat_rms"] = rms(fh[:, s0:], 0).tolist()
        return out

    n = y.shape[1]
    out = {"first_sample_scored": s0, "samples_scored": n - s0}
    for name, est in (("DMSO (rev. 8)", DMSO(F, G)), ("DMSO (ACC 2020)", DMSO2020(F, G))):
        est.initialize(y[:, 0])
        xt = y.copy()
        fh = np.zeros((2, n))
        for k in range(n - 1):
            xt[:, k + 1] = est.step(y[:, k], u[k])
            fh[:, k] = est.fhat
        out[name] = score(xt, fh)
    # what the robot computed on board (v9 DMSO, K = I, gamma = 0.05), mapped to the same
    # frame; the estimate logged in row k is its prediction for row k + 1
    tilt_ob, rate_ob = L["pitchm"] * DEG, L["gyhat"] * DEG
    onboard = np.vstack([-L["xhat"] - r * tilt_ob, -L["xhatdot"] - r * rate_ob, tilt_ob, rate_ob])
    xt = y.copy()
    xt[:, 1:] = onboard[:, :-1]
    out["on-board v9 DMSO"] = score(xt, None)
    out["measurement"] = {"gyro_consistency_dps": gyro_consistency(th)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["paper", "matrix", "gamma", "hardware"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rp = OUT / "results.json"
    if rp.exists():
        results.update(json.loads(rp.read_text()))
    steps = {"paper": paper, "matrix": matrix, "gamma": gamma, "hardware": hardware}
    for name, fn in steps.items():
        if args.only in (None, name):
            print(name)
            fn()
    rp.write_text(json.dumps(results, indent=1, default=float))
    print("wrote", rp)


if __name__ == "__main__":
    main()
