"""Test the plant model against the robot's logged balancing data.

    uv run python scripts/validate_plant.py           # everything
    uv run python scripts/validate_plant.py --quick   # fewer bootstrap and simulation repeats

The open question is the sign of the motor reaction torque in the pendulum moment
balance: ``+T`` (:mod:`twip.dynamics` ``"relative"``, the corrected model) or
``-T`` (``"rev8"``; the thesis revision-7 EOM ``"xdot"`` with the THESIS
parameters also has ``-T``).

Eliminating the motor torque from either model leaves an input-free relation
between the encoder position ``P = x + s r theta`` and the tilt,

    P_ddot = c1 theta_ddot + c2 theta           (linearized about upright)

that holds whatever the motor, its voltage, back-EMF, friction, deadzone or
saturation do, because the motor torque is internal to the robot.  For ``+T`` it
is the balance of angular momentum about the wheel contact point.  The two signs
predict very different ``c1`` and ``c2`` (:func:`model_table`), so the logs can
decide between them without a motor model.  ``s = +1`` is the physical encoder
kinematics (the encoders count wheel rotation relative to the body); ``s = -1``
is reported for completeness.

The same relation holds for the axle, ``x_ddot = a theta_ddot + c2 theta`` with
``a = c1 - s r``, and the accelerometer measures ``x_ddot`` directly, so it can be
tested without the encoder (whose gearbox backlash dominates small motions).

Sections (numbers follow the request in the PR thread):

1. inventory of the measured data,
2. input-free tests, each also run on simulated closed-loop runs of the +T and -T
   plants (with and without friction, backlash and rolling resistance) to show it
   recovers the right model:
   a. state-variable-filter least squares in the interior of the long runs
      (encoder, and accelerometer),
   b. the switch-on transients in integral form (encoder, and accelerometer),
   c. the tilt-to-encoder frequency response,
   d. closed-loop consistency: open-loop prediction of the switch-on transient
      from the logged voltages, and the robot's complementary filter in a -T loop,
3. input regression (B4, B2), the one-sample-delay question, ke from the back-EMF term,
4. the gains actually used, and closed-loop eigenvalues with them on each model,
5. 100 ms / 200 ms prediction from logged states and voltages,
6. motor-test data (ke, R; km is not identifiable from what was logged).

Section 8.4's hardware replay (``scripts/thesis_rev8.py --only hardware``) uses the
encoder frame established here.

Writes ``docs/thesis/generated/plant_validation.json`` and
``docs/thesis/figures/pv_*.pdf``.  Raw logs are read from the thesis folder
(:data:`twip.data.THESIS_ROOT`) and never copied.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, csd, filtfilt, welch

from twip.control import c2d_zoh, lqr
from twip.data import IMPLEMENTATION_COLUMNS, V8_COLUMNS, thesis_path
from twip.dynamics import accelerations, linear_model
from twip.legacy.observers import FirmwareDMSO
from twip.params import CORRECTED, THESIS
from twip.sensors import SensorConfig, SensorSuite

ROOT = Path(__file__).resolve().parents[1] / "docs" / "thesis"
FIG = ROOT / "figures"
GEN = ROOT / "generated"
DEG = math.pi / 180
R2D = 180 / math.pi
r = CORRECTED.r
COUNTS_PER_VOLT = 5904.5  # firmware: control (PWM counts) = volts * 5904.5

MODELS = {  # name -> (parameters, motor-coupling model of twip.dynamics)
    "+T": (CORRECTED, "relative"),
    "-T": (CORRECTED, "rev8"),
    "rev7": (THESIS, "xdot"),
}

# Firmware frame -> model frame.  The logged tilt has the model's sign (theta > 0
# leans the body toward -x), but the logged encoder position is mirrored:
# pos = -(x + r theta).  Established from the data in frame_evidence(); the input
# u has the model's sign.
POS_SIGN = -1.0
# MPU-9150 DLPF_CFG = 3 (42 Hz, 4.8 ms group delay) sampled at 200 Hz and read
# at 100 Hz: the gyro and accelerometer lag the encoder by about 7 ms.
IMU_DELAY = 0.007

QUICK = False
results: dict = {}


# ---------------------------------------------------------------------------
# model predictions


def model_table() -> dict:
    """c1, c2 of the input-free relation and the input gains, for each model and s."""
    out = {}
    for name, (p, be) in MODELS.items():
        A, B = linear_model(p, be)
        row = {"B2": B[1, 0], "B4": B[3, 0], "A": A.tolist(), "eig_open_loop": sorted(np.linalg.eigvals(A).real.tolist())}
        for s in (1, -1):
            c1 = (B[1, 0] + s * r * B[3, 0]) / B[3, 0]
            c2 = (A[1] + s * r * A[3] - c1 * A[3])[2]
            row[f"s={s:+d}"] = {"c1": c1, "c2": c2, "c2_over_c1": c2 / c1, "Bp": B[1, 0] + s * r * B[3, 0]}
        out[name] = row
    return out


def model_pdd(th: np.ndarray, w: np.ndarray, thdd: np.ndarray, model: str, s: float = 1.0) -> np.ndarray:
    """Nonlinear input-free relation: the encoder acceleration ``P_ddot`` that a model
    requires for a given tilt history, whatever the motor torque.  The model's
    accelerations are affine in the voltage, so the voltage that produces the
    measured ``theta_ddot`` is solved for and the matching ``x_ddot`` returned."""
    p, be = MODELS[model]
    out = np.empty_like(th)
    for k in range(len(th)):
        X = np.array([0.0, 0.0, th[k], w[k]])  # xdot only enters through T, which drops out
        x0, t0 = accelerations(X, 0.0, p, be)
        x1, t1 = accelerations(X, 1.0, p, be)
        v = (thdd[k] - t0) / (t1 - t0)
        out[k] = x0 + (x1 - x0) * v + s * r * thdd[k]
    return out


# ---------------------------------------------------------------------------
# 1. data


@dataclass
class Run:
    name: str
    firmware: str  # "v9", "v8" or "sim"
    t: np.ndarray  # s from switch-on
    dt: float  # median loop period
    P: np.ndarray  # m, encoder position in the model frame (x + r theta)
    th: np.ndarray  # rad, on-board filtered tilt (pitchc for v9, pitchk for v8)
    w: np.ndarray  # rad/s, gyro
    u: np.ndarray  # V, motor command trunc(control) / 5904.5, applied for the interval k -> k+1
    est: np.ndarray  # (4, n) on-board DMSO estimate in the firmware frame
    t_end: float  # end of the usable data (s)
    gain_eff: float  # average share of the command reaching the two motors
    K: np.ndarray | None = None  # gains recovered from the log (firmware frame, u = -K est)
    ax: np.ndarray | None = None  # g, accelerometer (body x, firmware sign: atan2(-ax, az) = tilt at rest)
    az: np.ndarray | None = None
    truth: dict = field(default_factory=dict)  # simulated runs only

    def mask(self, a: float, b: float) -> np.ndarray:
        return (self.t >= a) & (self.t <= min(b, self.t_end)) & (np.abs(self.th) < 15 * DEG)


def _parse(path: Path, ncol: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows with ``ncol`` numbers, and whether each line ended with a comma
    (v9 prints a trailing comma only in its idle loop)."""
    rows, trailing = [], []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            parts = line.strip().split(",")
            tr = parts[-1] == ""
            parts = [q for q in parts if q != ""]
            if len(parts) != ncol:
                continue
            try:
                rows.append([float(q) for q in parts])
            except ValueError:
                continue
            trailing.append(tr)
    return np.array(rows), np.array(trailing)


IMPL = ("Thesis", "Thesis Programs", "Implementation")
RUN_FILES = {  # file -> (firmware, usable end in s after switch-on)
    "implementationtest4.txt": ("v9", np.inf),
    "implementationtest5.txt": ("v9", np.inf),
    "implementationtest1.txt": ("v8", np.inf),
    "implementationtest2.txt": ("v8", 3.0),  # then a 37 Hz chatter at +-10 V and the robot falls
}


def load_runs(imu_delay: float = IMU_DELAY) -> list[Run]:
    runs = []
    for fname, (fw, t_end) in RUN_FILES.items():
        path = thesis_path(*IMPL, fname)
        if not path.exists():
            continue
        cols = IMPLEMENTATION_COLUMNS if fw == "v9" else V8_COLUMNS
        a, trailing = _parse(path, len(cols))
        L = {c: a[:, i] for i, c in enumerate(cols)}
        if fw == "v8":  # balancing rows print zeros in the gain columns
            bal = (L["lqr1"] == 0) & (L["lqr2"] == 0) & (L["lqr3"] == 0) & (L["lqr4"] == 0)
            gy_hat = L["gyhat"] * 57.2958 / 59.2958  # v8 logging typo (twip.data.fix_v8_gyhat)
        else:
            bal = ~trailing
            gy_hat = L["gyhat"]
        edges = np.flatnonzero(np.diff(np.r_[0, bal.astype(int), 0]))
        for s0, s1 in zip(edges[::2], edges[1::2]):
            if s1 - s0 < 100:
                continue
            sl = slice(s0, s1)
            dtk = L["dt"][sl]
            t = np.r_[0.0, np.cumsum(dtk[1:])]
            tilt = (L["pitchk"] if fw == "v8" else L["pitchc"])[sl] * DEG
            gyro = L["gy"][sl] * DEG
            ax, az = L["ax"][sl], L["az"][sl]
            if imu_delay:
                tilt, gyro, ax, az = (np.interp(t + imu_delay, t, q) for q in (tilt, gyro, ax, az))
            run = Run(
                name=fname.replace("implementationtest", "test").replace(".txt", ""),
                firmware=fw,
                t=t,
                dt=float(np.median(dtk)),
                P=POS_SIGN * L["x"][sl],
                th=tilt,
                w=gyro,
                u=np.trunc(L["control"][sl]) / COUNTS_PER_VOLT,
                est=np.vstack([L["xhat"][sl], L["xhatdot"][sl], L["pitchm"][sl] * DEG, gy_hat[sl] * DEG]),
                t_end=min(t_end, t[-1] - 0.2),
                gain_eff=0.9 if fw == "v9" else 0.965,  # Drive_Motor2: M1 gets -0.8 (v9) / -0.93 (v8) of the command
                ax=ax,
                az=az,
            )
            run.K = recover_gains(run)["K"]
            runs.append(run)
    return runs


def inventory() -> dict:
    """What measured data exist in the thesis folder, and what each can test."""
    items = [
        ("Thesis/Thesis Programs/Implementation/implementationtest4.txt", "v9 balancing log, DMSO in the loop (12.6 s)", "used: tasks 2-5"),
        ("Thesis/Thesis Programs/Implementation/implementationtest5.txt", "v9 balancing log, DMSO in the loop (21.2 s)", "used: tasks 2-5"),
        ("Thesis/Thesis Programs/Implementation/implementationtest1.txt", "v8 balancing log (3.1 s), pot gains logged", "used: tasks 2-5"),
        (
            "Thesis/Thesis Programs/Implementation/implementationtest2.txt",
            "v8 balancing log (3.9 s), ends in a 37 Hz chatter and fall",
            "used: tasks 2-5 (to 3.0 s)",
        ),
        ("Thesis/Thesis Programs/Implementation/implementationtest3.txt", "15-column log matching no firmware in twip_v1/", "not used"),
        (
            "Tests/LQR Tests",
            "lqrkalmantest*.txt: 26 balancing logs from Feb 2015 (v7), with at least 8 column layouts (lqrimport*.m); "
            + "most have pitch, x, xdot, dt, ax, az, gy, control",
            "not analysed: a candidate replication set",
        ),
        (
            "Tests/Motor Tests",
            "motortest1-13, 20khztest*, LoggedData.csv: PWM sweeps with speed/position/current, Sep 2014-Jan 2015, "
            + "on the 34:1, 48 CPR gearmotors that the v7 firmware (Feb 2015, 1920 counts/rev) replaced",
            "used: task 6 (motortest9)",
        ),
        ("Tests/Filtering Test", "filteringtest1-14: IMU filter comparisons, robot not balancing", "not relevant to the plant"),
        ("Tests/IMU Tests", "imutest1-5, accelerometerbiastest: static IMU data", "not relevant to the plant"),
        ("Tests/Allan Variance", "allandata.mat, filteringtest3.csv: static gyro noise", "not relevant to the plant"),
        (
            "newmotortest1.txt",
            "Feb 2015 balancing log on the new encoders (newmotortest1.mat: pitchc, pitchk, thetahat, x, xdot, dt, u); "
            + "no gyro or accelerometer",
            "not used",
        ),
        ("quicktest.txt", "Nov 2014 log, 10 columns; no column key", "not used"),
    ]
    out = []
    for rel, what, use in items:
        p = thesis_path(*rel.split("/"))
        entry = {"path": rel, "exists": p.exists(), "content": what, "use": use}
        if p.is_file():
            with open(p, encoding="latin-1") as fh:
                entry["lines"] = sum(1 for _ in fh)
        elif p.is_dir():
            entry["files"] = sum(1 for q in p.rglob("*") if q.is_file())
        out.append(entry)
    out.append(
        {
            "path": "(none)",
            "exists": False,
            "content": "no free-swing pendulum test, no wheels-off motor run with current and voltage logged together, "
            + "no back-driven ke test, no measured Ip or l",
            "use": "-",
        }
    )
    return {"items": out}


# ---------------------------------------------------------------------------
# filtering and regression helpers


def lowpass(x: np.ndarray, fc: float, dt: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth.  The same linear filter is applied to every signal,
    so any linear relation between the signals survives filtering (state-variable filter)."""
    b, a = butter(order, 2 * fc * dt)
    return filtfilt(b, a, x, padtype="odd", padlen=min(len(x) - 1, 150))


def d(x: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(x, dt)


def lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def block_index(n: int, block: int, rng) -> np.ndarray:
    """Indices of one moving-block bootstrap resample of a length-``n`` series."""
    block = min(block, n)
    starts = rng.integers(0, n - block + 1, math.ceil(n / block))
    return (starts[:, None] + np.arange(block)).ravel()[:n]


def ci(samples: np.ndarray) -> list[float]:
    return np.percentile(samples, [2.5, 97.5], axis=0).tolist()


# ---------------------------------------------------------------------------
# 2a. input-free relation: state-variable filter + least squares

HP_CUT = 0.2  # Hz, high-pass applied to every signal (removes gyro-integration drift)


def bandpass(x: np.ndarray, fc: float, dt: float) -> np.ndarray:
    """Zero-phase band-pass: 2nd-order Butterworth high-pass at HP_CUT and
    4th-order low-pass at ``fc``.  Applied identically to every signal, so linear
    relations between them, including derivative relations, are preserved."""
    b, a = butter(2, 2 * HP_CUT * dt, "high")
    y = filtfilt(b, a, x, padtype="odd", padlen=min(len(x) - 1, 300))
    return lowpass(y, fc, dt)


def gyro_tilt(run: Run) -> np.ndarray:
    """Tilt from the integrated gyro, started at the on-board filtered tilt at switch-on.
    The on-board pitchc/pitchk carry the accelerometer's (xddot - h thetaddot)/g error
    at low frequency, which biases c2; the gyro does not."""
    return run.th[0] + cumulative_trapezoid(run.w, run.t, initial=0.0)


def huber(X: np.ndarray, y: np.ndarray, k: float = 1.345, iters: int = 30) -> np.ndarray:
    """Huber M-estimate by IRLS (robust to the encoder jumps when the gears cross
    their backlash)."""
    c = lstsq(X, y)
    for _ in range(iters):
        e = y - X @ c
        sc = 1.4826 * np.median(np.abs(e - np.median(e))) + 1e-12
        wts = np.minimum(1.0, k * sc / np.maximum(np.abs(e), 1e-12))
        sw = np.sqrt(wts)
        c_new = lstsq(X * sw[:, None], y * sw)
        if np.allclose(c_new, c, rtol=1e-6, atol=1e-9):
            break
        c = c_new
    return c


def windows(run: Run) -> dict[str, tuple[float, float]]:
    """Regression windows for the filtered (SVF) fits.  Zero-phase filtering pads
    the record at its ends, which breaks the relation there, and the high-pass needs
    a few seconds to forget the edges, so only the interior of a long run is used.
    The switch-on transients go to the integral-form test instead (no filtering)."""
    if run.t_end - 1.0 - 3.0 < 3.0:
        return {}
    return {"steady": (3.0, run.t_end - 1.0)}


def svf_signals(run: Run, fc: float) -> dict[str, np.ndarray]:
    """Band-passed encoder, gyro tilt, gyro and accelerometer, and their derivatives."""
    dt = run.dt
    Pf = bandpass(run.P, fc, dt)
    wf = bandpass(run.w, fc, dt)
    out = {"Pd": d(Pf, dt), "th": bandpass(gyro_tilt(run), fc, dt), "w": wf, "thdd": d(wf, dt)}
    out["Pdd"] = d(out["Pd"], dt)
    if run.ax is not None:
        out["fax"] = bandpass(-9.81 * run.ax, fc, dt)  # = xddot - h thetaddot + g theta (linearized)
    return out


def fit_report(X: np.ndarray, y: np.ndarray, names: list[str], dt: float, n_boot: int, rng, robust: bool) -> tuple[dict, np.ndarray]:
    """Fit (least squares or Huber) with moving-block bootstrap CIs (0.5 s blocks)."""
    fit = huber if robust else lstsq
    c = fit(X, y)
    res = y - X @ c
    n, blk = len(y), round(0.5 / dt)
    nb = max(n_boot // 4, 50) if robust else n_boot  # IRLS is slower
    boot = np.empty((nb, X.shape[1]))
    for i in range(nb):
        idx = block_index(n, blk, rng)
        boot[i] = huber(X[idx], y[idx], iters=10) if robust else lstsq(X[idx], y[idx])
    out = {"n": n, "r2": float(1 - res.var() / y.var())}
    for i, k in enumerate(names):
        out[k] = float(c[i])
        out[k + "_ci"] = ci(boot[:, i])
    return out, boot


def svf_fit(run: Run, fc: float, window: str, free_terms: bool, n_boot: int, rng, robust: bool = False) -> dict | None:
    """``P_ddot = c1 theta_ddot + c2 theta`` (+ free ``P_dot``, ``theta_dot``) + const."""
    if window not in windows(run):
        return None
    a, b = windows(run)[window]
    m = run.mask(a, b)
    if m.sum() < 60:
        return None
    S = svf_signals(run, fc)
    cols, names = [S["thdd"], S["th"]], ["c1", "c2"]
    if free_terms:
        cols += [S["Pd"], S["w"]]
        names += ["Pdot", "thetadot"]
    cols.append(np.ones_like(S["th"]))
    names.append("const")
    out, boot = fit_report(np.column_stack(cols)[m], S["Pdd"][m], names, run.dt, n_boot, rng, robust)
    out["c2_over_c1"] = out["c2"] / out["c1"]
    out["c2_over_c1_ci"] = ci(boot[:, 1] / boot[:, 0])
    return out


def accel_fit(run: Run, fc: float, window: str, n_boot: int, rng) -> dict | None:
    """Encoder-free version: the accelerometer measures the IMU's acceleration, so
    ``-g ax = (a - h) theta_ddot + (c2 + g) theta`` with ``a = x_ddot / theta_ddot``
    of the input direction (c1 - r).  No encoder, backlash or kinematic sign involved."""
    if run.ax is None:
        return None
    a_, b_ = windows(run).get(window, (0, -1))
    m = run.mask(a_, b_)
    if m.sum() < 60:
        return None
    S = svf_signals(run, fc)
    X = np.column_stack([S["thdd"], S["th"], np.ones_like(S["th"])])[m]
    out, _ = fit_report(X, S["fax"][m], ["thdd_coef", "theta_coef", "const"], run.dt, n_boot, rng, False)
    return out


def svf_study(runs: list[Run], n_boot: int, rng) -> dict:
    out = {"svf": {}, "svf_huber": {}, "accel": {}}
    for run in runs:
        for fc in (2.0, 4.0, 8.0):
            for window in windows(run):
                key = f"{run.name}|{window}|{fc:g}Hz"
                for free in (True, False):
                    f = svf_fit(run, fc, window, free, n_boot, rng)
                    if f is not None:
                        out["svf"][key + ("|free" if free else "|nofree")] = f
                f = svf_fit(run, fc, window, False, n_boot, rng, robust=True)
                if f is not None:
                    out["svf_huber"][key] = f
                f = accel_fit(run, fc, window, n_boot, rng)
                if f is not None:
                    out["accel"][key] = f
    return out


# ---------------------------------------------------------------------------
# 2b. input-free relation in integral form over the switch-on transient


def dbl_int(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return cumulative_trapezoid(cumulative_trapezoid(y, t, initial=0.0), t, initial=0.0)


def _resid_bootstrap(X: np.ndarray, y: np.ndarray, blk: int, n_boot: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with a residual moving-block bootstrap (fixed design)."""
    c = lstsq(X, y)
    res = y - X @ c
    boots = np.array([lstsq(X, X @ c + res[block_index(len(y), blk, rng)]) for _ in range(n_boot)])
    return c, boots


def integral_test(run: Run, a: float, b: float, n_boot: int, rng, s: float = 1.0) -> dict | None:
    """Switch-on transient, integral form (no filtering, no differentiation of the encoder).

    Each model's nonlinear input-free relation turns the tilt history into a
    predicted encoder trajectory ``P(t)`` and axle trajectory ``x(t)``; these are
    compared with the encoder and with the double-integrated accelerometer
    (``x_ddot = -g (ax cos th + az sin th) + h (cos th th_ddot - sin th th_dot^2)``,
    which does not involve the encoder at all).  Tilt is the gyro integrated from the
    on-board tilt at ``a``.  Nuisance terms, fitted separately for each model: initial
    position and velocity, and a t^2 term (constant tilt or accelerometer offset).
    A free linear fit gives the data's own c1 (encoder) and a = c1 - s r (accelerometer)."""
    m = (run.t >= a) & (run.t <= min(b, run.t_end))
    if m.sum() < 30:
        return None
    t = run.t[m] - run.t[m][0]
    w = run.w[m]
    th = run.th[m][0] + cumulative_trapezoid(w, t, initial=0.0)
    thdd = d(run.w, run.dt)[m]
    N = np.column_stack([np.ones_like(t), t, t**2])
    h = SensorConfig().imu_height
    meas = {"encoder": run.P[m]}
    if run.ax is not None:
        g = 9.81
        c_, s_ = np.cos(th), np.sin(th)
        xdd_acc = -g * (run.ax[m] * c_ + run.az[m] * s_) + h * (c_ * thdd - s_ * w**2)
        meas["accelerometer"] = dbl_int(xdd_acc, t)
    out = {"window": [a, b], "P_range_mm": float(np.ptp(run.P[m]) * 1e3), "theta_range_deg": float(np.ptp(th) * R2D), "t": run.t[m]}
    blk = max(5, round(0.1 / run.dt))
    for key, y in meas.items():
        ss = s if key == "encoder" else 0.0  # the accelerometer sees the axle, x
        res = {"models": {}, "pred": {}, "meas": y}
        for name in MODELS:
            I2 = dbl_int(model_pdd(th, w, thdd, name, ss), t)
            c = lstsq(N, y - I2)
            pred = N @ c + I2
            res["models"][name] = {"rms_mm": float(np.sqrt(np.mean((y - pred) ** 2)) * 1e3)}
            res["pred"][name] = pred
        X = np.column_stack([N, th - th[0] - w[0] * t, dbl_int(th, t)])
        c, boots = _resid_bootstrap(X, y, blk, n_boot, rng)
        coef = "c1" if key == "encoder" else "a"
        res["free_fit"] = {
            coef: c[3],
            coef + "_ci": ci(boots[:, 3]),
            "c2": c[4],
            "c2_ci": ci(boots[:, 4]),
            "rms_mm": float((y - X @ c).std() * 1e3),
        }
        out[key] = res
    return out


# ---------------------------------------------------------------------------
# 2c. tilt-to-encoder frequency response


def transfer(run: Run, a: float, b: float, nperseg: int = 128) -> dict | None:
    """H(f) = P/theta from gyro-to-encoder cross spectra.  The input-free relation
    predicts H = c1 - c2 / (2 pi f)^2, real, for any closed loop and any motor."""
    m = run.mask(a, b)
    if m.sum() < 2 * nperseg:
        return None
    fs = 1 / run.dt
    kw = {"fs": fs, "nperseg": nperseg, "detrend": "linear"}
    f, Sww = welch(run.w[m], **kw)
    _, Swp = csd(run.w[m], run.P[m], **kw)
    _, Spp = welch(run.P[m], **kw)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = 2j * np.pi * f * Swp / Sww
        coh = np.abs(Swp) ** 2 / (Sww * Spp)
    return {"f": f, "H": H, "coh": coh, "n": int(m.sum())}


def transfer_summary(tf: dict, mt: dict, band: tuple[float, float] = (5.0, 15.0)) -> dict:
    """Measured vs predicted H(f).  ``ReH_band``: coherence-weighted mean of Re H over
    ``band`` (dominated by c1); each model's prediction uses the same weights.  ``|H|``
    at 1 and 2 Hz, where the gravity term matters (and where stick-slip and backlash
    dominate the real robot's small motions)."""
    f, H, coh = tf["f"], tf["H"], tf["coh"]
    sel = (f >= band[0]) & (f <= band[1]) & np.isfinite(H)
    wts = coh[sel] / np.maximum(1 - coh[sel], 0.05)
    om2 = (2 * np.pi * f[sel]) ** 2
    out = {"ReH_band": float(np.sum(wts * H.real[sel]) / np.sum(wts)), "band_Hz": list(band), "mean_coherence_band": float(coh[sel].mean())}
    for fq in (1.0, 2.0):
        i = int(np.argmin(np.abs(f - fq)))
        out[f"H_at_{fq:g}Hz"] = {
            "f": float(f[i]),
            "re": float(H[i].real),
            "im": float(H[i].imag),
            "abs": float(abs(H[i])),
            "coherence": float(coh[i]),
        }
    out["models"] = {}
    for name, row in mt.items():
        c1, c2 = row["s=+1"]["c1"], row["s=+1"]["c2"]
        out["models"][name] = {
            "ReH_band": float(np.sum(wts * (c1 - c2 / om2)) / np.sum(wts)),
            "H_at_1Hz": c1 - c2 / (2 * np.pi) ** 2,
            "H_at_2Hz": c1 - c2 / (4 * np.pi) ** 2,
        }
    return out


# ---------------------------------------------------------------------------
# simulated closed-loop runs (method validation)


def sim_gains(model: str, dt: float = 0.01) -> np.ndarray:
    """A stabilizing LQR for the model (the logged v9 gains do not stabilize -T),
    returned in the firmware frame."""
    A, B = linear_model(*MODELS[model])
    K = lqr(A, B, np.diag([20.0, 1.0, 50.0, 1.0]), 1.0).ravel()
    return np.array([POS_SIGN * K[0], POS_SIGN * K[1], K[2], K[3]])


def simulate(
    model: str,
    seed: int,
    T: float = 15.0,
    theta0_deg: float = 6.0,
    nuisance: bool = False,
    dt: float = 0.01,
    alpha: float = 0.99,
    K: np.ndarray | None = None,
) -> Run:
    """The v9 loop on a simulated plant: sensors (twip.sensors), complementary
    filter, the firmware DMSO, LQR, saturation at 10 V, truncation to PWM counts and
    M1 at 0.8 of the command.  Excitation is internal (a 0.3 V dither added to the
    motor voltage), so the input-free relation holds exactly in the plant.

    ``nuisance`` adds what the real logs suggest: gearbox Coulomb friction (0.6 V
    equivalent), 2 deg of gearbox backlash between the encoder (motor side) and the
    wheel, and floor rolling resistance (0.02 of the weight, external).

    ``alpha`` is the complementary-filter weight (0.99 on the robot); ``alpha = 1``
    integrates the gyro alone.  A -T plant needs ``alpha = 1`` to balance at all
    (see :func:`complementary_filter_check`)."""
    p, be = MODELS[model]
    rng = np.random.default_rng(seed)
    cfg = SensorConfig()
    sens = SensorSuite(cfg, p, rng)
    K = sim_gains(model, dt) if K is None else np.asarray(K, float)
    dmso = FirmwareDMSO()
    gain_eff = 0.9
    Vc, eps_w = (0.6, 0.5) if nuisance else (0.0, 1.0)
    backlash = 2 * DEG if nuisance else 0.0
    mu_rr = 0.02 if nuisance else 0.0
    weight = (p.Mp + 2 * p.Mw) * p.g
    beta = 2 * p.Mw + 2 * p.Iw / p.r**2 + p.Mp
    rel = {"relative": 1.0, "xdot": 0.0, "rev8": -1.0}[be]  # coefficient of thetadot in w_rel

    def torque(X, v):
        w_rel = X[1] / p.r + rel * X[3]
        return 2 * p.km / p.R * (v - p.ke * w_rel), w_rel

    def f(X, v):
        _, w_rel = torque(X, v)
        v_tot = v - Vc * math.tanh(w_rel / eps_w)  # internal: acts through the motor torque
        xdd, thdd = accelerations(X, v_tot, p, be)
        if mu_rr:
            F = -mu_rr * weight * math.tanh(X[1] / 0.01)  # external force on x (rolling resistance)
            c = math.cos(X[2])
            M = np.array([[beta, -p.Mp * p.l * c], [-p.Mp * p.l * c, p.Ip + p.Mp * p.l**2]])
            dx, dth = np.linalg.solve(M, [F, 0.0])
            xdd, thdd = xdd + dx, thdd + dth
        return np.array([X[1], xdd, X[3], thdd])

    N = round(T / dt)
    X = np.array([0.0, 0.0, theta0_deg * DEG, 0.0])
    pitchc = theta0_deg
    # converge the DMSO while the robot is held still at theta0 (idle loop, drive = 0)
    for _ in range(200):
        dmso.step(0.0, 0.0, pitchc, 0.0, 0.0)
    drive, v, pos_prev, dither = 0, 0.0, None, 0.0
    log = {k: np.zeros(N) for k in ("P", "th", "w", "u", "x", "theta", "ax", "az")}
    est = np.zeros((4, N))
    a_d = math.exp(-dt / 0.05)
    for k in range(N):
        xdd, thdd = f(X, v)[[1, 3]]
        m = sens.measure(X, xdd, thdd, dt)
        T_m, _ = torque(X, v)
        phi_rel = X[0] / p.r + X[2] + 0.5 * backlash * math.tanh(T_m / 0.02)  # motor side of the gearbox
        counts = round(phi_rel * cfg.counts_per_rev / (2 * math.pi))
        pos = POS_SIGN * counts * cfg.meters_per_count  # firmware frame
        xdot = 0.0 if pos_prev is None else (pos - pos_prev) / dt
        pos_prev = pos
        pitchc = alpha * (pitchc + m.gy * dt) + (1 - alpha) * math.atan2(-m.ax, m.az) * R2D
        e, _ = dmso.step(pos, xdot, pitchc, m.gy, drive)
        control = float(np.clip(-K @ e, -10, 10)) * COUNTS_PER_VOLT
        drive = math.trunc(control)
        dither = a_d * dither + math.sqrt(1 - a_d**2) * 0.3 * rng.standard_normal()
        v = gain_eff * drive / COUNTS_PER_VOLT + dither
        log["P"][k], log["th"][k], log["w"][k], log["u"][k] = POS_SIGN * pos, pitchc * DEG, m.gy * DEG, drive / COUNTS_PER_VOLT
        log["ax"][k], log["az"][k] = m.ax, m.az
        log["x"][k], log["theta"][k] = X[0], X[2]
        est[:, k] = e
        for _ in range(4):  # RK4, 4 substeps
            h = dt / 4
            k1 = f(X, v)
            k2 = f(X + h / 2 * k1, v)
            k3 = f(X + h / 2 * k2, v)
            k4 = f(X + h * k3, v)
            X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if abs(X[2]) > 0.8:
            raise RuntimeError(f"simulated {model} run fell at t={k * dt:.2f}")
    t = np.arange(N) * dt
    return Run(
        name=f"sim {model}{' +nuisance' if nuisance else ''} #{seed}",
        firmware="sim",
        t=t,
        dt=dt,
        P=log["P"],
        th=log["th"],
        w=log["w"],
        u=log["u"],
        est=est,
        t_end=T - 0.2,
        gain_eff=gain_eff,
        K=K,
        ax=log["ax"],
        az=log["az"],
        truth={"model": model, "x": log["x"], "theta": log["theta"]},
    )


# ---------------------------------------------------------------------------
# 4. gains actually used


def recover_gains(run: Run) -> dict:
    """Regress the logged control on the logged DMSO estimate (the firmware computes
    control = -K est, clipped at 10 V); the fit should be exact to print precision."""
    c = run.u  # trunc() changes it by < 1 count
    ok = np.abs(c) < 9.99
    X = np.column_stack([run.est.T, np.ones(len(c))])[ok]
    k = lstsq(X, c[ok])
    res = c[ok] - X @ k
    return {
        "K": -k[:4],
        "const_V": k[4],
        "rms_V": float(res.std()),
        "max_abs_V": float(np.abs(res).max()),
        "r2": float(1 - res.var() / c[ok].var()),
        "n": int(ok.sum()),
    }


def closed_loop_eigs(
    model: str, K_fw: np.ndarray, dt: float, gain: float, observer: str = "none", s: float = 1.0, pos_sign: float = POS_SIGN
) -> np.ndarray:
    """Discrete closed-loop eigenvalues (z) of the firmware loop on a linear model.

    Measurements in the firmware frame: pos = pos_sign (x + s r theta) (encoder),
    xdot = backward difference of pos, tilt, gyro.  ``observer``: ``"none"``
    (u = -K y), ``"v9"`` (the firmware DMSO with W = 0: est = F est_prev + G u_prev
    + (y - est_prev)) or ``"v8"`` (v8 MSO: est = F est_prev + G u_prev + Kmso (y - est_prev)).
    Plant input: gain * u, ZOH over dt."""
    A, B = linear_model(*MODELS[model])
    Phi, Gam = c2d_zoh(A, B * gain, dt)
    prow = pos_sign * np.array([1.0, 0.0, s * r, 0.0, 0.0])
    M = np.zeros((4, 5))  # y = M [x; pos_prev]
    M[0] = prow
    M[1] = (prow - np.r_[0, 0, 0, 0, 1.0]) / dt
    M[2, 2] = M[3, 3] = 1.0
    K = np.asarray(K_fw, float)
    if observer == "none":
        Acl = np.zeros((5, 5))
        Acl[:4, :4] = Phi
        Acl[:4] -= np.outer(Gam[:, 0], K @ M)
        Acl[4] = prow
        return np.linalg.eigvals(Acl)
    if observer == "v9":
        F, G, Ko = FirmwareDMSO.F, FirmwareDMSO.G, np.eye(4)
    else:
        # v8 filters.ino, including its F[0][1] = .130 (for .0130)
        F = np.array([[1, 0.130, 0.0002, 0], [0, 0.9987, 0.0311, 0.0002], [0, -0.000008, 1.0034, 0.0130], [0, -0.0012, 0.5163, 1.0034]])
        G = np.array([0.0001, 0.0159, 0.0001, 0.0153])
        Ko = np.array([[0.9902, 0, 0, 0], [0, 0.9999, 0, 0], [0, 0, 0.9168, 0.0003], [0, 0.0001, 0.3436, 0.5863]])
    # z = [x(4), pos_prev, est_prev(4), u_prev];  est = (F - Ko) est_prev + G u_prev + Ko M z[:5]
    E = np.zeros((4, 10))
    E[:, :5] = Ko @ M
    E[:, 5:9] = F - Ko
    E[:, 9] = G
    U = -K @ E
    Acl = np.zeros((10, 10))
    Acl[:4, :4] = Phi
    Acl[:4] += np.outer(Gam[:, 0], U)
    Acl[4, :5] = prow
    Acl[5:9] = E
    Acl[9] = U
    return np.linalg.eigvals(Acl)


def eig_summary(z: np.ndarray, dt: float) -> dict:
    """Spectral radius and the least stable modes as continuous-time equivalents."""
    z = z[np.abs(z) > 1e-9]
    s_eq = np.log(z.astype(complex)) / dt
    order = np.argsort(-np.abs(z))
    modes = []
    for i in order:
        if s_eq[i].imag < -1e-9:
            continue
        modes.append([round(float(s_eq[i].real), 2), round(float(abs(s_eq[i].imag) / (2 * np.pi)), 2)])
    return {"max_abs_z": float(np.abs(z).max()), "stable": bool(np.abs(z).max() < 1), "modes_sigma_per_s_and_Hz": modes[:4]}


def gain_margin_scan(model: str, K_fw: np.ndarray, dt: float, gain: float, observer: str) -> list[float] | None:
    """Range of input-gain multipliers (x0.1 .. x10 on km/R) over which the loop is stable."""
    gs = np.logspace(-1, 1, 81)
    ok = [g for g in gs if np.abs(closed_loop_eigs(model, K_fw, dt, gain * g, observer)).max() < 1]
    return [float(min(ok)), float(max(ok))] if ok else None


def run_ending(run: Run) -> dict:
    """How a run ended: drift of the base (log-linear fit to the encoder position over
    the last 2.5 s of usable data), final tilt and the dominant gyro frequency, plus the
    part after ``t_end`` if one was cut off."""
    m = (run.t >= max(0.5, run.t_end - 2.5)) & (run.t <= run.t_end)
    t, P = run.t[m], run.P[m] - run.P[m][0]
    f, S = welch(run.w[m], 1 / run.dt, nperseg=min(m.sum(), 128))
    out = {
        "t_end_s": float(run.t[-1]),
        "P_change_mm": float(P[-1] * 1e3),
        "theta_end_deg": float(run.th[m][-1] * R2D),
        "gyro_peak_Hz": float(f[np.argmax(S[1:]) + 1]),
    }
    big = np.abs(P) > 0.002
    if big.sum() > 20:
        out["P_growth_rate_per_s"] = float(np.polyfit(t[big], np.log(np.abs(P[big])), 1)[0])
    tail = run.t > run.t_end + 0.2
    if tail.sum() > 16:
        f2, S2 = welch(run.w[tail], 1 / run.dt, nperseg=min(tail.sum(), 64))
        out["after_t_end"] = {
            "duration_s": float(run.t[-1] - run.t_end),
            "gyro_peak_Hz": float(f2[np.argmax(S2[1:]) + 1]),
            "theta_min_max_deg": [float(run.th[tail].min() * R2D), float(run.th[tail].max() * R2D)],
            "u_saturated_frac": float(np.mean(np.abs(run.u[tail]) > 9.99)),
        }
    return out


def eigen_study(runs: list[Run]) -> dict:
    out = {}
    for run in runs:
        rec = recover_gains(run)
        K = rec["K"]
        entry = {
            "gains_fw_frame": K.tolist(),
            "gains_model_frame": [POS_SIGN * K[0], POS_SIGN * K[1], K[2], K[3]],
            "fit": {k: v for k, v in rec.items() if k != "K"},
            "dt": run.dt,
            "input_share": run.gain_eff,
            "models": {},
            "ending": run_ending(run),
        }
        obs_list = ["none", "v9"] if run.firmware == "v9" else ["none", "v8"]
        for name in MODELS:
            e = {}
            for obs in obs_list:
                e[f"observer={obs}"] = eig_summary(closed_loop_eigs(name, K, run.dt, run.gain_eff, obs), run.dt)
                e[f"observer={obs}|stable_input_gain_range"] = gain_margin_scan(name, K, run.dt, run.gain_eff, obs)
            K0 = K.copy()
            K0[:2] = 0
            e["tilt_loop_only (K1=K2=0)"] = eig_summary(closed_loop_eigs(name, K0, run.dt, run.gain_eff, "none"), run.dt)
            e["if_encoder_not_mirrored"] = eig_summary(closed_loop_eigs(name, K, run.dt, run.gain_eff, "none", pos_sign=-POS_SIGN), run.dt)
            entry["models"][name] = e
        out[run.name] = entry
    return out


# ---------------------------------------------------------------------------
# 2d. closed-loop consistency


def complementary_filter_check(seeds: int = 2) -> dict:
    """At low frequency a balancing robot accelerates at x_ddot = c2 theta, so the
    accelerometer tilt reads theta (1 + c2/g): +0.43 theta for +T but -2.95 theta for
    -T.  With the robot's complementary filter (alpha = 0.99) in the loop, a -T robot
    is fed a tilt of the wrong sign at low frequency.  Simulate both plants with three
    LQR weightings, with the robot's filter and with the gyro alone (alpha = 1)."""
    mt = model_table()
    out = {"accel_tilt_ratio_low_freq": {k: 1 + v["s=+1"]["c2"] / 9.81 for k, v in mt.items()}, "runs": {}}
    weights = {"Q=diag(20,1,50,1)": [20, 1, 50, 1], "Q=diag(1,1,100,1)": [1, 1, 100, 1], "Q=diag(100,10,500,10)": [100, 10, 500, 10]}
    for name in ("+T", "-T"):
        A, B = linear_model(*MODELS[name])
        for wname, q in weights.items():
            Km = lqr(A, B, np.diag(q), 1.0).ravel()
            K = np.array([POS_SIGN * Km[0], POS_SIGN * Km[1], Km[2], Km[3]])
            for alpha in (0.99, 1.0):
                res = []
                for seed in range(seeds):
                    try:
                        simulate(name, 100 + seed, T=10.0, theta0_deg=3.0, alpha=alpha, K=K)
                        res.append(None)
                    except RuntimeError as e:
                        res.append(float(str(e).split("t=")[1]))
                out["runs"][f"{name}|{wname}|alpha={alpha}"] = {"fell_at_s": res}
    return out


def _rk4_open_loop(X: np.ndarray, v: float, p, be, dt: float, sub: int = 10) -> np.ndarray:
    def f(Y):
        xdd, thdd = accelerations(Y, v, p, be)
        return np.array([Y[1], xdd, Y[3], thdd])

    h = dt / sub
    for _ in range(sub):
        k1 = f(X)
        k2 = f(X + h / 2 * k1)
        k3 = f(X + h / 2 * k2)
        k4 = f(X + h * k3)
        X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X


def startup_test(runs: list[Run]) -> dict:
    """Open-loop prediction of the first 50-150 ms after switch-on, from rest at the
    logged lean with the logged (saturated) voltages, on each nonlinear model.  The
    robot may still have been held by hand for part of this, so this is a
    consistency check rather than a clean test."""
    out = {}
    for run in runs:
        if run.firmware != "v9":
            continue
        e = {"theta0_deg": float(run.th[0] * R2D), "u_first_V": run.u[:6].tolist(), "measured": {}, "models": {}}
        for h in (5, 10, 15):
            e["measured"][f"{h * 10}ms"] = {
                "dtheta_deg": float((run.th[h] - run.th[0]) * R2D),
                "dP_mm": float((run.P[h] - run.P[0]) * 1e3),
                "w_degps": float(run.w[h] * R2D),
            }
        for name, (p, be) in MODELS.items():
            X = np.array([0.0, 0.0, run.th[0], 0.0])
            res = {}
            for k in range(15):
                X = _rk4_open_loop(X, run.u[k] * run.gain_eff, p, be, run.dt)
                if k + 1 in (5, 10, 15):
                    res[f"{(k + 1) * 10}ms"] = {
                        "dtheta_deg": float((X[2] - run.th[0]) * R2D),
                        "dP_mm": float((X[0] + r * (X[2] - run.th[0])) * 1e3),
                        "w_degps": float(X[3] * R2D),
                    }
            e["models"][name] = res
        out[run.name] = e
    return out


# ---------------------------------------------------------------------------
# 3. input regression


def input_regression(runs: list[Run], n_boot: int, rng) -> dict:
    """theta_ddot and P_ddot on [P_dot, theta, theta_dot, u, 1] over the switch-on
    transient (0.05-1.5 s; the voltage spans -10..+10 V there), unfiltered.  Accelerations
    are forward differences k -> k+1; ``lag`` 0 regresses on u[k] (the firmware applies
    u[k] right after computing it), lag 1 on u[k-1].  P_dot is the logged backward
    difference, theta the gyro integrated from the switch-on tilt.

    Model predictions are per volt at the motors; the logged command reaches them
    scaled by ``input_share`` (v9: M1 gets 0.8 of it, M2 1.0).  In these coordinates
    the +T back-EMF acts on P_dot alone (w_rel = P_dot / r), so ke = -a_P r / B4 and
    the theta_dot coefficient is 0; -T and rev7 put part of it on theta_dot."""
    out = {"models": {}, "runs": {}}
    for name, (p, be) in MODELS.items():
        A, B = linear_model(p, be)

        def row(v):  # x = [x, xd, th, w] with xd = Pd - r w (s = +1)
            return {"Pdot": v[1], "theta": v[2], "thetadot": v[3] - r * v[1]}

        out["models"][name] = {
            "thdd": {**row(A[3]), "u": B[3, 0]},
            "Pdd": {**row(A[1] + r * A[3]), "u": B[1, 0] + r * B[3, 0]},
            "Bp_over_B4": (B[1, 0] + r * B[3, 0]) / B[3, 0],
            "ke_ratio": -A[3, 1] * r / B[3, 0],
        }
    for run in runs:
        m = (run.t >= 0.05) & (run.t <= min(1.5, run.t_end))
        idx = np.flatnonzero(m)
        idx = idx[(idx >= 2) & (idx < len(run.t) - 1)]
        th = run.th[0] + cumulative_trapezoid(run.w, run.t, initial=0.0)
        Pd = np.r_[0.0, np.diff(run.P)] / run.dt
        y_th = (run.w[idx + 1] - run.w[idx]) / run.dt
        y_p = (Pd[idx + 1] - Pd[idx]) / run.dt
        e = {"n": len(idx), "input_share": run.gain_eff}
        for lag in (0, 1):
            X = np.column_stack([Pd[idx], th[idx], run.w[idx], run.u[idx - lag], np.ones(len(idx))])
            names = ["Pdot", "theta", "thetadot", "u", "const"]
            fth, bth = fit_report(X, y_th, names, run.dt, n_boot, rng, False)
            fp, bp = fit_report(X, y_p, names, run.dt, n_boot, rng, False)
            cu = lstsq(X[:, [0, 1, 2, 4]], X[:, 3])
            r2u = 1 - np.var(X[:, 3] - X[:, [0, 1, 2, 4]] @ cu) / np.var(X[:, 3])
            e[f"lag={lag}"] = {
                "thdd": fth,
                "Pdd": fp,
                "B4_per_logged_V": fth["u"],
                "B4_per_motor_V": fth["u"] / run.gain_eff,
                "Bp_per_logged_V": fp["u"],
                "Bp_over_B4": fp["u"] / fth["u"],
                "Bp_over_B4_ci": ci(bp[:, 3] / bth[:, 3]),
                "B2_over_B4 (s=+1)": fp["u"] / fth["u"] - r,
                "ke_implied_if_+T": -fth["Pdot"] * r / fth["u"],
                "ke_implied_ci": ci(-bth[:, 0] * r / bth[:, 3]),
                "thetadot_over_Pdot_coef": fth["thetadot"] / fth["Pdot"],
                "u_r2_on_states": float(r2u),
                "u_vif": float(1 / max(1 - r2u, 1e-9)),
            }
        X = np.column_stack([Pd[idx], th[idx], run.w[idx], run.u[idx], run.u[idx - 1], np.ones(len(idx))])
        c = lstsq(X, y_th)
        e["both_lags_thdd_u_coefs"] = {"u[k]": c[3], "u[k-1]": c[4]}
        out["runs"][run.name] = e
    return out


# ---------------------------------------------------------------------------
# 5. multi-step prediction


def prediction_study(runs: list[Run]) -> dict:
    """Predict tilt and encoder position 100 and 200 ms ahead from the logged state
    (P, a local-polynomial P_dot, on-board tilt, gyro) and the logged voltages, with
    each linear model; compare with holding the state and with constant velocity."""
    from scipy.signal import savgol_filter

    out = {}
    for run in runs:
        dt = run.dt
        Pd = savgol_filter(run.P, 7, 2, deriv=1, delta=dt)
        x = np.vstack([run.P - r * run.th, Pd - r * run.w, run.th, run.w])
        segs = {"transient": (0.1, 1.5)}
        if windows(run):
            segs["steady"] = windows(run)["steady"]
        e = {}
        for seg, (a, b) in segs.items():
            for horizon in (0.1, 0.2):
                hN = round(horizon / dt)
                ks = np.flatnonzero((run.t >= a) & (run.t + horizon <= min(b, run.t_end)))
                if len(ks) < 10:
                    continue
                res = {"n": len(ks), "steps": hN}
                tgt_th, tgt_P = run.th[ks + hN], run.P[ks + hN]

                def score(pth, pP, tgt_th=tgt_th, tgt_P=tgt_P):
                    return {
                        "theta_rms_deg": float(np.sqrt(np.mean((pth - tgt_th) ** 2)) * R2D),
                        "P_rms_mm": float(np.sqrt(np.mean((pP - tgt_P) ** 2)) * 1e3),
                    }

                res["hold"] = score(run.th[ks], run.P[ks])
                res["const_velocity"] = score(run.th[ks] + run.w[ks] * hN * dt, run.P[ks] + Pd[ks] * hN * dt)
                for name, model in MODELS.items():
                    A, B = linear_model(*model)
                    F, G = c2d_zoh(A, B * run.gain_eff, dt)
                    pth, pP = np.empty(len(ks)), np.empty(len(ks))
                    for j, k in enumerate(ks):
                        z = x[:, k].copy()
                        for i in range(hN):
                            z = F @ z + G[:, 0] * run.u[k + i]
                        pth[j], pP[j] = z[2], z[0] + r * z[2]
                    res[name] = score(pth, pP)
                e[f"{seg}|{round(horizon * 1000)}ms"] = res
        out[run.name] = e
    return out


# ---------------------------------------------------------------------------
# 6. motor tests


def _kv_log(path: Path) -> dict[str, np.ndarray]:
    recs, cur = [], {}
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                v = float(v)
            except ValueError:
                continue
            k = k.strip()
            if k == "Time" and cur:
                recs.append(cur)
                cur = {}
            cur[k] = v
    recs.append(cur)
    keys = sorted({k for q in recs for k in q})
    return {k: np.array([q.get(k, np.nan) for q in recs]) for k in keys}


def motor_tests(v_batt: float = 12.0) -> dict:
    """``motortest9.txt`` (Sep 2014): slow PWM sweeps (-400..400, the stock VNH5019
    library's full scale) with the firmware's speed (RPM) and current (mA) for both
    motors.  Steady armature equation ``V = R i + ke w + V_dz sign(w)`` with
    ``V = control / 400 * v_batt``.  These motors predate the March 2015 runs: the
    firmware's encoder scale changes from 1633 counts/rev (34:1, 48 CPR, the 2014
    parts list) to 1920 (30:1, 64 CPR, thesis Table 4.2) at v7 (Feb 2015)."""
    path = thesis_path("Tests", "Motor Tests", "motortest9.txt")
    if not path.exists():
        return {}
    L = _kv_log(path)
    out = {"file": "Tests/Motor Tests/motortest9.txt", "v_batt_assumed": v_batt, "motors": {}}
    for mname in ("M1", "M2"):
        c, w, i = L[f"{mname} control"], L[f"{mname} speed"] * 2 * np.pi / 60, L[f"{mname} current"] / 1000
        ok = np.isfinite(c) & np.isfinite(w) & np.isfinite(i) & (np.abs(w) > 1.0)
        V = c[ok] / 400 * v_batt
        sgn = np.sign(w[ok])
        cf = lstsq(np.column_stack([i[ok] * sgn, w[ok], sgn]), V)  # the current sense reads magnitude
        cf2 = lstsq(np.column_stack([w[ok], sgn]), V)
        moving = np.isfinite(w) & (np.abs(w) > 1.0)
        out["motors"][mname] = {
            "R_ohm": cf[0],
            "ke_Vs_per_rad": cf[1],
            "deadzone_V": cf[2],
            "ke_ignoring_current": cf2[0],
            "deadzone_ignoring_current_V": cf2[1],
            "max_speed_rad_s": float(np.nanmax(np.abs(w))),
            "breakaway_control_counts": float(np.nanmin(np.abs(c[moving]))) if moving.any() else None,
            "n": int(ok.sum()),
        }
    out["km_note"] = (
        "no torque, or acceleration with a known inertia, was logged: km is not identifiable (ideal DC motor in SI units: km = ke)"
    )
    return out


def startup_gain_fit(run: Run, n: int = 10) -> dict:
    """Scale each model's motor torque (km, i.e. input gain and back-EMF torque
    together) to best match the logged tilt over the first ``n`` samples after
    switch-on, then report the encoder error at that scale.  A model whose input
    direction is wrong cannot match both."""
    out = {}
    th_m = run.th[1 : n + 1]
    P_m = run.P[1 : n + 1] - run.P[0]
    for name, (p0, be) in MODELS.items():
        best = None
        for g in np.logspace(-1, 1.5, 151):
            p = p0.scaled(km=g)
            X = np.array([0.0, 0.0, run.th[0], 0.0])
            th, P = np.empty(n), np.empty(n)
            for k in range(n):
                X = _rk4_open_loop(X, run.u[k] * run.gain_eff, p, be, run.dt, sub=5)
                th[k], P[k] = X[2], X[0] + r * (X[2] - run.th[0])
            e_th = float(np.sqrt(np.mean((th - th_m) ** 2)) * R2D)
            if best is None or e_th < best["theta_rms_deg"]:
                best = {
                    "km_scale": float(g),
                    "theta_rms_deg": e_th,
                    "P_rms_mm": float(np.sqrt(np.mean((P - P_m) ** 2)) * 1e3),
                    "B4_per_motor_V": float(g * linear_model(p0, be)[1][3, 0]),
                }
        out[name] = best
    return out


# ---------------------------------------------------------------------------
# figures


MODEL_STYLE = {"+T": {"color": "C0"}, "-T": {"color": "C3"}, "rev7": {"color": "C2", "ls": "--"}}


def fig_transient(integral: dict) -> None:
    """Top: encoder trajectory and each model's prediction from the tilt history.
    Bottom: residual of the double-integrated accelerometer against each model (its
    t^2 nuisance, an accelerometer or tilt offset, dwarfs the trajectory itself)."""
    keys = [k for k in ("test4", "test5") if k in integral]
    if not keys:
        return
    fig, axs = plt.subplots(2, len(keys), figsize=(6.5, 4.6), squeeze=False, sharex=True)
    for j, name in enumerate(keys):
        it = integral[name]["[0.1, 1.5]"]
        t = it["t"]
        ax = axs[0, j]
        ax.plot(t, it["encoder"]["meas"] * 1e3, "k", lw=1.8, label="measured")
        for mname, pred in it["encoder"]["pred"].items():
            ax.plot(t, pred * 1e3, lw=1.0, label=f"{mname} ({it['encoder']['models'][mname]['rms_mm']:.1f} mm rms)", **MODEL_STYLE[mname])
        ax.set(title=name, ylabel="Encoder $P$ (mm)")
        ax.legend(fontsize=6)
        if "accelerometer" in it:
            ax = axs[1, j]
            q = it["accelerometer"]
            for mname, pred in q["pred"].items():
                ax.plot(
                    t, (q["meas"] - pred) * 1e3, lw=1.0, label=f"{mname} ({q['models'][mname]['rms_mm']:.1f} mm rms)", **MODEL_STYLE[mname]
                )
            ax.axhline(0, color="k", lw=0.5)
            ax.set(xlabel="Time after switch-on (s)", ylabel="Accelerometer residual (mm)")
            ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "pv_transient.pdf")
    plt.close(fig)


def fig_startup(runs: list[Run]) -> None:
    v9 = [rn for rn in runs if rn.firmware == "v9"]
    if not v9:
        return
    fig, axs = plt.subplots(2, len(v9), figsize=(6.5, 4.2), squeeze=False)
    for j, run in enumerate(v9):
        n = 16
        t = run.t[:n] * 1e3
        axs[0, j].plot(t, run.th[:n] * R2D, "k", lw=1.8, label="measured")
        axs[1, j].plot(t, (run.P[:n] - run.P[0]) * 1e3, "k", lw=1.8)
        for name, (p, be) in MODELS.items():
            X = np.array([0.0, 0.0, run.th[0], 0.0])
            th, P = [X[2]], [0.0]
            for k in range(n - 1):
                X = _rk4_open_loop(X, run.u[k] * run.gain_eff, p, be, run.dt)
                th.append(X[2])
                P.append(X[0] + r * (X[2] - run.th[0]))
            axs[0, j].plot(t, np.array(th) * R2D, lw=1.0, label=name, **MODEL_STYLE[name])
            axs[1, j].plot(t, np.array(P) * 1e3, lw=1.0, **MODEL_STYLE[name])
        axs[0, j].set(title=f"{run.name}: -10 V from {run.th[0] * R2D:.0f} deg", ylabel="Tilt (deg)")
        axs[1, j].set(xlabel="Time after switch-on (ms)", ylabel="Encoder $P$ (mm)")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "pv_startup.pdf")
    plt.close(fig)


def fig_frequency(tfs: dict, mt: dict) -> None:
    keys = list(tfs)
    if not keys:
        return
    fig, axs = plt.subplots(2, len(keys), figsize=(3.3 * len(keys), 4.6), squeeze=False, sharex=True)
    for j, name in enumerate(keys):
        tf = tfs[name]
        f = tf["f"][1:]
        H, coh = tf["H"][1:], tf["coh"][1:]
        axs[0, j].semilogx(f, H.real, "k.", ms=3, label="measured Re $H$")
        om2 = (2 * np.pi * f) ** 2
        for mname, row in mt.items():
            c1, c2 = row["s=+1"]["c1"], row["s=+1"]["c2"]
            axs[0, j].semilogx(f, c1 - c2 / om2, lw=1.0, label=mname, **MODEL_STYLE[mname])
        axs[0, j].set(ylim=(-0.6, 2.5), title=name)
        axs[1, j].semilogx(f, coh, "k", lw=0.8)
        axs[1, j].set(xlabel="Frequency (Hz)", ylim=(0, 1))
    axs[0, 0].set_ylabel(r"$H = P/\theta$ (m/rad)")
    axs[1, 0].set_ylabel("Coherence")
    axs[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "pv_frequency.pdf")
    plt.close(fig)


def fig_coefficients(rows: list[tuple[str, float, list[float], str]], mt: dict) -> None:
    """Forest plot of the inertial coefficient: c1 (encoder) or a + r (accelerometer, s = +1)."""
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(rows) + 1.2))
    for i, (lab, val, lohi, kind) in enumerate(rows):
        y = len(rows) - i
        mk = {"data": "ko", "sim +T": "C0s", "sim -T": "C3s"}[kind]
        ax.errorbar(val, y, xerr=[[val - lohi[0]], [lohi[1] - val]] if lohi else None, fmt=mk, ms=4, capsize=2, lw=0.8)
    for mname, row in mt.items():
        ax.axvline(row["s=+1"]["c1"], lw=1.0, label=f"{mname}: {row['s=+1']['c1']:.2f}", **MODEL_STYLE[mname])
    ax.set_yticks(range(len(rows), 0, -1))
    ax.set_yticklabels([q[0] for q in rows], fontsize=7)
    ax.set(xlabel=r"$c_1$ (encoder) or $a + r$ (accelerometer)", xlim=(-0.3, 1.4))
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "pv_coefficients.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------


def _clean(o):
    """JSON-safe copy without the plotted arrays."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if k not in ("t", "meas", "pred", "f", "H", "coh")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def frame_evidence(runs: list[Run]) -> dict:
    """At switch-on (u = -10 V) the logged position and the tilt moved in opposite
    senses.  Every model has B_p / B4 = c1 > 0 (wheel and body accelerate in the same
    sense in the model frame), so the logged position is mirrored: pos = -(x + r theta)."""
    out = {}
    for run in runs:
        if run.firmware != "v9":
            continue
        out[run.name] = {
            "u_V": float(run.u[0]),
            "dpos_logged_mm_100ms": float((run.P[10] - run.P[0]) / POS_SIGN * 1e3),
            "dtheta_deg_100ms": float((run.th[10] - run.th[0]) * R2D),
        }
    return out


def summarize(res: dict, runs: list[Run]) -> dict:
    """The headline numbers, one entry per test (data only; simulated runs are in the
    detailed sections)."""
    mt = res["model_table"]
    pred_c1 = {k: round(v["s=+1"]["c1"], 3) for k, v in mt.items()}
    pred_a = {k: round(v["B2"] / v["B4"], 3) for k, v in mt.items()}
    S = {
        "predicted": {
            "c1 (s=+1)": pred_c1,
            "c1 (s=-1)": {k: round(v["s=-1"]["c1"], 3) for k, v in mt.items()},
            "c2": {k: round(v["s=+1"]["c2"], 2) for k, v in mt.items()},
            "a = x_ddot/theta_ddot of the input": pred_a,
            "B4": {k: round(v["B4"], 2) for k, v in mt.items()},
        },
        "data": {},
    }
    I, sv = res["input_free"]["integral"], res["input_free"]["svf"]
    for rn in runs:
        e = {}
        it = I.get(rn.name, {}).get("[0.1, 0.8]")
        if it:
            e["transient_rms_mm_encoder"] = {k: round(v["rms_mm"], 1) for k, v in it["encoder"]["models"].items()}
            e["transient_c1_encoder"] = [
                round(it["encoder"]["free_fit"]["c1"], 3),
                [round(x, 3) for x in it["encoder"]["free_fit"]["c1_ci"]],
            ]
            if "accelerometer" in it:
                e["transient_rms_mm_accelerometer"] = {k: round(v["rms_mm"], 1) for k, v in it["accelerometer"]["models"].items()}
                e["transient_a_accelerometer"] = [
                    round(it["accelerometer"]["free_fit"]["a"], 3),
                    [round(x, 3) for x in it["accelerometer"]["free_fit"]["a_ci"]],
                ]
        q = sv["accel"].get(f"{rn.name}|steady|8Hz")
        if q:
            e["steady_accel_thdd_coef_8Hz (predict a - h)"] = [round(q["thdd_coef"], 3), [round(x, 3) for x in q["thdd_coef_ci"]]]
            e["steady_accel_theta_coef_8Hz (predict c2 + g)"] = [round(q["theta_coef"], 2), [round(x, 2) for x in q["theta_coef_ci"]]]
        q = sv["svf"].get(f"{rn.name}|steady|8Hz|nofree")
        if q:
            e["steady_svf_8Hz_encoder"] = {
                "c1": [round(q["c1"], 3), [round(x, 3) for x in q["c1_ci"]]],
                "c2": [round(q["c2"], 2), [round(x, 2) for x in q["c2_ci"]]],
                "r2": round(q["r2"], 2),
            }
        tf = res["input_free"]["transfer"].get(rn.name)
        if tf:
            e["ReH_5_15Hz"] = round(tf["ReH_band"], 3)
            e["absH_1Hz_2Hz"] = [round(tf["H_at_1Hz"]["abs"], 3), round(tf["H_at_2Hz"]["abs"], 3)]
        st = res["input_free"]["startup"].get(rn.name)
        if st:
            e["startup_dtheta_100ms_deg"] = {
                "measured": round(st["measured"]["100ms"]["dtheta_deg"], 1),
                **{k: round(v["100ms"]["dtheta_deg"], 1) for k, v in st["models"].items()},
            }
            e["startup_km_scale_fit"] = {
                k: {"km_scale": round(v["km_scale"], 2), "theta_rms_deg": round(v["theta_rms_deg"], 2), "P_rms_mm": round(v["P_rms_mm"], 1)}
                for k, v in st["km_scale_fit"].items()
            }
        eg = res["eigen"].get(rn.name)
        if eg:
            obs = "v9" if rn.firmware == "v9" else "v8"
            e["gains_fw_frame"] = [round(x, 3) for x in eg["gains_fw_frame"]]
            e["least_stable_mode_per_s"] = {k: v[f"observer={obs}"]["modes_sigma_per_s_and_Hz"][0] for k, v in eg["models"].items()}
            e["ending"] = eg["ending"]
        S["data"][rn.name] = e
    return S


def main() -> None:
    global QUICK
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer bootstrap repeats and simulated runs")
    args = ap.parse_args()
    QUICK = args.quick
    n_boot = 100 if QUICK else 500
    n_seeds = 1 if QUICK else 2
    rng = np.random.default_rng(2026)
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    mt = model_table()
    results["model_table"] = mt
    results["settings"] = {"imu_delay_s": IMU_DELAY, "pos_sign": POS_SIGN, "hp_cut_Hz": HP_CUT, "n_boot": n_boot, "sim_seeds": n_seeds}
    print("1. inventory")
    results["inventory"] = inventory()
    runs = load_runs()
    print("   runs:", ", ".join(f"{rn.name} ({rn.firmware}, {rn.t[-1]:.1f} s)" for rn in runs))
    results["frame_evidence"] = frame_evidence(runs)

    print("2. input-free tests: simulated validation runs")
    sims, fallen = [], []

    def sim_ok(model: str, seed: int, **kw) -> Run:
        """Next seed that balances for the whole run (a -T plant with friction sometimes falls)."""
        for sd in range(seed, seed + 20):
            try:
                return simulate(model, sd, **kw)
            except RuntimeError as e:
                fallen.append(f"{model} {kw} seed {sd}: {e}")
        raise RuntimeError(f"no balancing {model} run for {kw}")

    for i in range(n_seeds):
        sims += [
            sim_ok("+T", 1 + 20 * i),
            sim_ok("+T", 101 + 20 * i, nuisance=True),
            sim_ok("-T", 1 + 20 * i, alpha=1.0),
            sim_ok("-T", 101 + 20 * i, alpha=1.0, nuisance=True),
        ]
    results["simulated_runs"] = {"used": [rn.name for rn in sims], "fell_and_skipped": fallen}
    allruns = sims + runs
    print("   SVF least squares")
    sv = svf_study(allruns, n_boot, rng)
    print("   integral form over the switch-on transient")
    integral = {}
    for rn in allruns:
        integral[rn.name] = {}
        for a, b in ((0.1, 0.8), (0.1, 1.5)):
            it = integral_test(rn, a, b, n_boot, rng)
            if it is not None:
                integral[rn.name][f"[{a}, {b}]"] = it
    print("   frequency response")
    tfs, tfsum = {}, {}
    for rn in allruns:
        w = windows(rn)
        if "steady" in w:
            tf = transfer(rn, *w["steady"])
            if tf is not None:
                tfs[rn.name] = tf
                tfsum[rn.name] = transfer_summary(tf, mt)
    print("   closed-loop consistency")
    cfc = complementary_filter_check(seeds=n_seeds)
    st = startup_test(runs)
    for rn in runs:
        if rn.name in st:
            st[rn.name]["km_scale_fit"] = startup_gain_fit(rn)
    # sensitivity: no IMU delay correction
    nodelay = {}
    for rn in load_runs(imu_delay=0.0):
        it = integral_test(rn, 0.1, 0.8, n_boot, rng)
        if it is not None:
            nodelay[rn.name] = {
                key: {"rms_mm": {k: v["rms_mm"] for k, v in it[key]["models"].items()}, "free_fit": it[key]["free_fit"]}
                for key in ("encoder", "accelerometer")
                if key in it
            }
    results["input_free"] = {
        "svf": sv,
        "integral": integral,
        "integral_without_imu_delay_[0.1, 0.8]": nodelay,
        "transfer": tfsum,
        "complementary_filter": cfc,
        "startup": st,
    }

    print("3. input regression")
    results["input_regression"] = input_regression(runs, n_boot, rng)
    print("4. gains and closed-loop eigenvalues")
    results["eigen"] = eigen_study(runs)
    print("5. multi-step prediction")
    results["prediction"] = prediction_study(runs)
    print("6. motor tests")
    results["motor_tests"] = motor_tests()

    print("figures")
    fig_transient(integral)
    fig_startup(runs)
    fig_frequency({k: v for k, v in tfs.items() if not k.startswith("sim") or k.endswith("#1")}, mt)
    rows = []
    for rn in sims[:4] + runs:  # one seed of each simulated case
        kind = "data" if rn.firmware != "sim" else ("sim +T" if rn.truth["model"] == "+T" else "sim -T")
        it = integral.get(rn.name, {}).get("[0.1, 0.8]")
        if it:
            ff = it["encoder"]["free_fit"]
            rows.append((f"{rn.name}: transient, encoder", ff["c1"], ff["c1_ci"], kind))
            if "accelerometer" in it:
                ff = it["accelerometer"]["free_fit"]
                rows.append((f"{rn.name}: transient, accelerometer", ff["a"] + r, [ff["a_ci"][0] + r, ff["a_ci"][1] + r], kind))
        q = sv["accel"].get(f"{rn.name}|steady|8Hz")
        if q:
            rows.append(
                (
                    f"{rn.name}: steady, accelerometer 8 Hz",
                    q["thdd_coef"] + r + SensorConfig().imu_height,
                    [q["thdd_coef_ci"][0] + r + SensorConfig().imu_height, q["thdd_coef_ci"][1] + r + SensorConfig().imu_height],
                    kind,
                )
            )
        q = sv["svf"].get(f"{rn.name}|steady|8Hz|nofree")
        if q:
            rows.append((f"{rn.name}: steady, encoder 8 Hz", q["c1"], q["c1_ci"], kind))
    fig_coefficients(rows, mt)

    out = {"summary": summarize(results, runs), **results}
    (GEN / "plant_validation.json").write_text(json.dumps(_clean(out), indent=1))
    print("wrote", GEN / "plant_validation.json")


if __name__ == "__main__":
    main()
