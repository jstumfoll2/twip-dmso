"""Nonlinear and linearized equations of motion of the TWIP.

State vector used throughout: ``[x, xdot, theta, thetadot]`` with the motor
voltage ``v`` as input.

Angle conventions
-----------------
The MATLAB code uses two conventions for the tilt angle and switches between
them from folder to folder:

* **down** (``EOM Test``, ``Extra control``, ``Tests/LQR Tests``):
  ``theta = 0`` is hanging straight down and ``theta = pi`` is upright.
  These files write ``sind(theta*180/pi)`` / ``cosd(theta*180/pi)``.
* **up** (``no noise ...`` and ``Noise with uncertainty`` DMSO sims):
  ``theta = 0`` is upright.  These files write ``cos(-pi/2-theta)`` for
  sin and ``sin(-pi/2-theta)`` for cos, which is exactly
  ``theta_down = theta_up + pi``.

:func:`twip_nonlinear` takes an explicit ``convention`` argument so both can be
reproduced from one implementation.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .params import NOMINAL, RobotParams

Convention = Literal["up", "down"]


def twip_nonlinear(
    X: np.ndarray,
    v: float,
    p: RobotParams = NOMINAL,
    convention: Convention = "up",
    extra: Callable[[np.ndarray], tuple[float, float]] | None = None,
) -> np.ndarray:
    """Time derivative of ``[x, xdot, theta, thetadot]`` for motor voltage ``v``.

    Port of the uncommented ``% old equations`` block of ``twipnonlinear.m``
    (the symbolic-toolbox output, not the hand derivation).

    ``extra`` optionally returns additive ``(xddot, thetaddot)`` terms. It is
    used to reproduce the unmodeled terms in ``twipnonlinear_uncert.m``.
    """
    x, xdot, theta, thetadot = X[0], X[1], X[2], X[3]
    th = theta if convention == "down" else theta + np.pi
    s, c = np.sin(th), np.cos(th)

    Mp, Mw, r, Ip, Iw = p.Mp, p.Mw, p.r, p.Ip, p.Iw
    g, l, km, ke, R = p.g, p.l, p.km, p.ke, p.R

    den = (
        2 * Ip * Iw
        + Mp**2 * l**2 * r**2
        + 2 * Iw * Mp * l**2
        + Ip * Mp * r**2
        + 2 * Ip * Mw * r**2
        + 2 * Mp * Mw * l**2 * r**2
        - Mp**2 * l**2 * r**2 * c**2
    )
    xddot = (
        R * s * Mp**2 * l**3 * r**2 * thetadot**2
        + R * g * c * s * Mp**2 * l**2 * r**2
        + 2 * km * v * Mp * l**2 * r
        - 2 * ke * km * xdot * Mp * l**2
        + Ip * R * s * Mp * l * r**2 * thetadot**2
        + 2 * km * v * c * Mp * l * r**2
        - 2 * ke * km * xdot * c * Mp * l * r
        + 2 * Ip * km * v * r
        - 2 * Ip * ke * km * xdot
    ) / (R * den)
    thetaddot = -(
        2 * Mp * km * r**3 * v
        + 4 * Mw * km * r**3 * v
        - 4 * Iw * ke * km * xdot
        + 4 * Iw * km * r * v
        - 2 * Mp * ke * km * r**2 * xdot
        - 4 * Mw * ke * km * r**2 * xdot
        + 2 * Mp * km * l * r**2 * v * c
        + Mp**2 * R * g * l * r**3 * s
        + 2 * Iw * Mp * R * g * l * r * s
        + Mp**2 * R * l**2 * r**3 * thetadot**2 * c * s
        - 2 * Mp * ke * km * l * r * xdot * c
        + 2 * Mp * Mw * R * g * l * r**3 * s
    ) / (R * r * den)

    if extra is not None:
        dx2, dx4 = extra(X)
        xddot += dx2
        thetaddot += dx4

    return np.array([xdot, xddot, thetadot, thetaddot])


def uncert_extra_terms(X: np.ndarray) -> tuple[float, float]:
    """Unmodeled terms added in ``twipnonlinear_uncert.m``:
    ``0.5*sin(x)*xdot`` on xddot and ``0.25*xdot^2`` on thetaddot."""
    return 0.5 * np.sin(X[0]) * X[1], 0.25 * X[1] ** 2


def linear_model(p: RobotParams = NOMINAL) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time model ``xdot = A x + B v`` linearized about upright.

    Port of the ``% Nonlinear dc motors, linearized EOM variables`` block.
    The angle is measured from upright (theta = 0 upright).
    """
    Mp, Mw, r, Ip, Iw = p.Mp, p.Mw, p.r, p.Ip, p.Iw
    g, l, km, ke, R, b = p.g, p.l, p.km, p.ke, p.R, p.b

    beta = 2 * Iw / r**2 + 2 * Mw + Mp
    alpha = Ip * beta + 2 * Mp * l**2 * (Mw + Iw / r**2)
    A22 = -b + 2 * km * ke * (Mp * l * r - Ip - Mp * l**2) / (R * r**2 * alpha)
    A23 = Mp**2 * g * l**2 / alpha
    A42 = -b + 2 * km * ke * (r * beta - Mp * l) / (R * r**2 * alpha)
    A43 = Mp * g * l * beta / alpha
    B1 = 2 * km * (Ip + Mp * l**2 - Mp * l * r) / (R * r * alpha)
    B2 = 2 * km * (Mp * l - r * beta) / (R * r * alpha)

    A = np.array(
        [
            [0, 1, 0, 0],
            [0, A22, A23, 0],
            [0, 0, 0, 1],
            [0, A42, A43, 0],
        ],
        dtype=float,
    )
    B = np.array([[0], [B1], [0], [B2]], dtype=float)
    return A, B
