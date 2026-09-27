"""Ports of the post-processing scripts that ran on logged robot data.

* :func:`replay_filters` - ``Tests/Filtering Test/filteringtest.m``
* :func:`replay_implementation` - ``Thesis Programs/Implementation/lqrkalmantest1v2.m``
* :func:`replay_firmware_v9` - the robot's own DMSO (``twip_v9/filters.ino``) on the same logs
* :func:`allan` - ``Tests/Allan Variance/allan.m`` (vectorized)
* :func:`gauss_markov_bias_variance` - the 140-sample binning in ``allanprocess.m``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control import c2d_zoh
from .dynamics import linear_model
from .estimators import DMSO, AngleBiasKalman, DMSO2, FirmwareDMSO, LinearKalman, complementary_filter
from .params import RobotParams

DEG = np.pi / 180


def replay_filters(d: dict[str, np.ndarray], alpha: float = 0.98, firmware_bug: bool = True) -> dict[str, np.ndarray]:
    """Re-run the robot's three tilt filters offline on raw ``gy, ax, az, dt``.

    With ``firmware_bug=True`` the DMSO output matches the ``pitchmso`` column
    logged by ``twip_v4`` (see :class:`twip.estimators.DMSO2`).
    """
    n = len(d["dt"])
    kf = AngleBiasKalman()
    mso = DMSO2(firmware_bug=firmware_bug)
    out = {k: np.zeros(n) for k in ("thetac", "thetak", "bias", "mso", "msodot", "fhat1", "fhat2")}
    pc = 0.0
    for i in range(n):
        gy, ax, az, dt = d["gy"][i], d["ax"][i], d["az"][i], d["dt"][i]
        x, f = mso.step(gy, ax, az, dt)
        out["mso"][i], out["msodot"][i] = x
        out["fhat1"][i], out["fhat2"][i] = f
        pc = complementary_filter(pc, gy, ax, az, dt, alpha)
        out["thetac"][i] = pc
        out["thetak"][i], out["bias"][i] = kf.step(gy, ax, az, dt)
    out["t"] = np.concatenate(([0.0], np.cumsum(d["dt"][:-1])))
    return out


# Control counts -> volts.  twip_v9/controllers.ino: "5904.5 PWM input/V"
# (about 65535 PWM counts / 11.1 V battery).
CONTROL_COUNTS_PER_VOLT = 5904.5


@dataclass
class ImplementationReplay:
    t: np.ndarray
    accpitch: np.ndarray  # deg
    comp: np.ndarray  # deg
    kalman: np.ndarray  # (4, n+1) SI units
    xhatmso: np.ndarray  # (4, n+1) SI units
    fhat: np.ndarray  # (2, n)
    control_volts: np.ndarray


def replay_implementation(
    log: dict[str, np.ndarray],
    params: RobotParams | None = None,
    reproduce_R_bug: bool = True,
) -> ImplementationReplay:
    """Offline 4-state Kalman + DMSO on a hardware LQR run (``lqrkalmantest1v2.m``).

    ``reproduce_R_bug``: in the MATLAB script, ``R = .1`` for an unused
    2-state filter overwrites the 4x4 Kalman ``R`` before the loop.  The
    4-state Kalman then adds 0.1 to every element of ``H P H'``.  Set False to
    use the intended diagonal R.

    The script also sets ``km = .1154`` and ``ke = .0036`` (rounded values),
    which is the default here.
    """
    p = params or RobotParams(km=0.1154, ke=0.0036)
    dt = log["dt"]
    n = len(dt) - 1
    A, B = linear_model(p)
    F, G = c2d_zoh(A, B, float(np.mean(dt)))

    Q = np.diag([0.0012, 0.095, 0.001, 0.12])
    R = 0.1 if reproduce_R_bug else np.diag([1e-5, 1e-5, 4e-6, 0.0012 * np.pi / 180])
    kf = LinearKalman(F, G, Q, R, P0=0.01 * np.eye(4))
    mso = DMSO(F, G, 0.25 * 2 * np.eye(4), gamma=0.05, sigma=0.0)

    u = log["control"] / CONTROL_COUNTS_PER_VOLT
    kal = np.zeros((4, n + 1))
    xm = np.zeros((4, n + 1))
    fhat = np.zeros((2, n))
    accpitch = np.zeros(n)
    comp = np.zeros(n + 1)
    comp[0] = log["pitchm"][0]
    for i in range(n):
        accpitch[i] = np.arctan2(-log["ax"][i], log["az"][i]) * 57.296
        comp[i + 1] = 0.99 * (comp[i] + log["gy"][i] * dt[i]) + 0.01 * accpitch[i]
        y = np.array([log["x"][i], log["xdot"][i], accpitch[i] * 0.01745, log["gy"][i] * 0.01745])
        kal[:, i + 1] = kf.step(y, u[i])
        xm[:, i + 1], fhat[:, i] = mso.step(y, u[i])
        mso.update_weights(y - xm[:, i + 1])  # this script uses the *post-update* estimate
    t = np.concatenate(([0.0], np.cumsum(dt[:-1])))
    return ImplementationReplay(t, accpitch, comp, kal, xm, fhat, u)


def replay_firmware_v9(log: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Recompute the robot's on-board DMSO from a ``twip_v9`` log.

    Reproduces the logged ``xhat``, ``xhatdot``, ``pitchm`` and ``gyhat`` columns to
    print precision.  The motor command used at step k is the one logged at step k-1.
    """
    mso = FirmwareDMSO()
    n = len(log["dt"])
    est = np.zeros((4, n))
    fhat = np.zeros((2, n))
    drive = 0.0
    for i in range(n):
        est[:, i], fhat[:, i] = mso.step(log["x"][i], log["xdot"][i], log["pitchc"][i], log["gy"][i], drive)
        drive = np.trunc(log["control"][i])  # `int drive = LQR(...)`
    return {
        "xhat": est[0], "xhatdot": est[1], "pitchm": est[2] * 57.2958, "gyhat": est[3] * 57.2958,
        "fhat1": fhat[0], "fhat2": fhat[1],
    }


@dataclass
class AllanResult:
    tau: np.ndarray  # cluster times actually used (m / rate)
    sig: np.ndarray  # std of non-overlapping cluster means
    sig2: np.ndarray  # Allan deviation, non-overlapping
    osig: np.ndarray  # Allan deviation, overlapping
    sigerr: np.ndarray
    sig2err: np.ndarray
    osigerr: np.ndarray


def allan(freq: np.ndarray, rate: float, tau: np.ndarray) -> AllanResult:
    """Allan deviation of a constant-rate series.  Vectorized port of ``allan.m``.

    That file is the widely circulated ``allan.m`` by M.A. Hopcroft; its
    loops are O(n * len(tau)), which is slow for the 1.9M-sample gyro log.
    """
    freq = np.asarray(freq, float)
    n = len(freq)
    m = np.floor(np.asarray(tau) * rate).astype(int)
    c = np.concatenate(([0.0], np.cumsum(freq)))
    out = {k: np.zeros(len(m)) for k in ("sig", "sig2", "osig", "sigerr", "sig2err", "osigerr")}
    for j, mj in enumerate(m):
        if mj < 1:  # tau shorter than one sample; MATLAB yields NaN here
            for v in out.values():
                v[j] = np.nan
            continue
        D = (c[mj:] - c[: n - mj + 1]) / mj  # all length-mj window means, len n-mj+1
        Dn = D[::mj]
        out["sig"][j] = np.std(Dn, ddof=1) if len(Dn) > 1 else np.nan
        out["sig2"][j] = np.sqrt(0.5 * np.mean(np.diff(Dn) ** 2)) if len(Dn) > 1 else np.nan
        z1 = D[mj : n + 1 - mj]
        z2 = D[: n + 1 - 2 * mj]
        out["osig"][j] = np.sqrt(np.sum((z1 - z2) ** 2) / (n + 1 - 2 * mj) / 2)
        out["sigerr"][j] = out["sig"][j] / np.sqrt(n / mj)
        out["sig2err"][j] = out["sig2"][j] / np.sqrt(n / mj)
        out["osigerr"][j] = out["osig"][j] / np.sqrt(n - mj)
    return AllanResult(tau=m / rate, **out)


def gauss_markov_bias_variance(signal: np.ndarray, bin_size: int = 140) -> float:
    """Variance of the mean-removed signal averaged in ``bin_size`` bins.

    ``allanprocess.m`` uses this as the walking-bias variance (the ``vargbias``
    and ``varabias`` values in the simulations).  The MATLAB inner loop
    ``for j = j:j+140`` sums 141 samples, with neighbouring bins overlapping by
    one, and divides by 140.  This port uses clean 140-sample bins, so expect
    small differences.
    """
    s = np.asarray(signal, float)
    s = s - s.mean()
    nb = len(s) // bin_size
    return float(np.var(s[: nb * bin_size].reshape(nb, bin_size).mean(axis=1), ddof=1))
