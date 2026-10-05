"""Equations of motion of the TWIP (corrected baseline).

State ``X = [x, xdot, theta, thetadot]``: wheel-axle position (m), its rate,
body tilt (rad, 0 = upright, same sign convention as the thesis and the robot's
``pitch``), and tilt rate.  Input ``v`` is the voltage applied to *each* motor.

The body CG sits at ``(x - l sin(theta), l cos(theta))`` relative to the axle, so
``theta > 0`` leans the body backward (counterclockwise, CG toward ``-x``).
Lagrangian of a planar TWIP with both wheels lumped together::

    [ beta          -Mp l cos(th) ] [xdd ]   [ T/r - Mp l sin(th) thd^2 ]
    [ -Mp l cos(th)  Ip + Mp l^2  ] [thdd] = [ +T  + Mp g l sin(th)      ]

with ``beta = 2 Mw + 2 Iw / r^2 + Mp`` and total motor torque
``T = 2 km / R * (v - ke * w_rel) - 2 b w_rel``.  ``Ip`` is the body inertia
about its CG.

Motor coupling.  Rolling forward turns the wheels clockwise at ``xdot / r``,
while ``theta`` is counterclockwise, so the motor shaft turns relative to its
stator (fixed to the body) at ``w_rel = xdot / r + thetadot``.  The virtual work
of the motor torque through that relative angle gives the generalized forces
``T / r`` on ``x`` and ``+T`` on ``theta``: driving the wheels forward pitches the
body backward.  This is the only sign that conserves angular momentum about the
wheel contact point, where every external force acts or (at upright) passes
through (tested).

The thesis pendulum moment balance (revision 7, and the first revision-8 code)
used ``-T`` on ``theta``.  Three motor-coupling models are available:

* ``"relative"`` (default): ``+T``, ``w_rel = xdot/r + thetadot``.  Physically
  consistent.
* ``"xdot"``: the thesis EOM, ``-T`` and ``xdot / r`` only.  Reproduces
  :mod:`twip.legacy.dynamics` exactly (tested to 1e-12).
* ``"rev8"``: ``-T`` with ``w_rel = xdot/r - thetadot``, the truth model behind the
  first revision-8 results, kept so those numbers stay reproducible.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .params import CORRECTED, RobotParams

BackEMF = Literal["relative", "xdot", "rev8"]
Disturbance = Callable[[np.ndarray], "tuple[float, float] | np.ndarray"]

# (reaction sign s on theta, coefficient of thetadot in w_rel) for each motor model:
# generalized forces are [T/r, s*T] and w_rel = xdot/r + c*thetadot.  Energy is
# consistent (motor power = T * w_rel) only when c = s.
_COUPLING = {"relative": (1.0, 1.0), "xdot": (-1.0, 0.0), "rev8": (-1.0, -1.0)}


def _torque(xdot: float, thetadot: float, v: float, p: RobotParams, back_emf: BackEMF) -> float:
    w_rel = xdot / p.r + _COUPLING[back_emf][1] * thetadot
    return 2 * p.km / p.R * (v - p.ke * w_rel) - 2 * p.b * w_rel


def accelerations(
    X: np.ndarray,
    v: float,
    p: RobotParams = CORRECTED,
    back_emf: BackEMF = "relative",
    disturbance: Disturbance | None = None,
) -> tuple[float, float]:
    """``(xddot, thetaddot)`` for state ``X`` and motor voltage ``v``.

    ``disturbance(X)`` optionally returns additive ``(xddot, thetaddot)`` terms,
    for unmodeled dynamics or external forces.
    """
    _, xdot, th, thd = X[0], X[1], X[2], X[3]
    s, c = np.sin(th), np.cos(th)
    Mp, l, r = p.Mp, p.l, p.r
    beta = 2 * p.Mw + 2 * p.Iw / r**2 + Mp
    m11, m12, m22 = beta, -Mp * l * c, p.Ip + Mp * l**2
    T = _torque(xdot, thd, v, p, back_emf)
    f1 = T / r - Mp * l * s * thd**2
    f2 = _COUPLING[back_emf][0] * T + Mp * p.g * l * s
    det = m11 * m22 - m12 * m12
    xdd = (m22 * f1 - m12 * f2) / det
    thdd = (m11 * f2 - m12 * f1) / det
    if disturbance is not None:
        d = disturbance(X)
        xdd += d[0]
        thdd += d[1]
    return xdd, thdd


def derivative(
    X: np.ndarray,
    v: float,
    p: RobotParams = CORRECTED,
    back_emf: BackEMF = "relative",
    disturbance: Disturbance | None = None,
) -> np.ndarray:
    """Time derivative of ``X``."""
    xdd, thdd = accelerations(X, v, p, back_emf, disturbance)
    return np.array([X[1], xdd, X[3], thdd])


def linear_model(p: RobotParams = CORRECTED, back_emf: BackEMF = "relative") -> tuple[np.ndarray, np.ndarray]:
    """Continuous ``xdot = A x + B v`` linearized about upright (theta = 0).

    Exact linearization of :func:`derivative` (tested against its Jacobian).
    """
    Mp, l, r = p.Mp, p.l, p.r
    beta = 2 * p.Mw + 2 * p.Iw / r**2 + Mp
    Minv = np.linalg.inv(np.array([[beta, -Mp * l], [-Mp * l, p.Ip + Mp * l**2]]))
    sign, rel = _COUPLING[back_emf]
    # T = cv*v - cw*w_rel, with w_rel = xdot/r + rel*thetadot
    cv = 2 * p.km / p.R
    cw = 2 * p.km * p.ke / p.R + 2 * p.b
    dT = np.array([0.0, -cw / r, 0.0, -cw * rel])  # dT/d[x, xdot, th, thd]
    # generalized forces [T/r, sign*T + Mp g l th], linearized
    Q = np.vstack([dT / r, sign * dT + np.array([0.0, 0.0, Mp * p.g * l, 0.0])])
    acc = Minv @ Q
    A = np.zeros((4, 4))
    A[0, 1] = A[2, 3] = 1.0
    A[1], A[3] = acc
    b = Minv @ np.array([cv / r, sign * cv])
    B = np.array([[0.0], [b[0]], [0.0], [b[1]]])
    return A, B
