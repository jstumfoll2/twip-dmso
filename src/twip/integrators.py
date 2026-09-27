"""Fixed-step integrator (port of ``RK4.m``)."""

from __future__ import annotations

from typing import Callable

import numpy as np


def rk4(
    f: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    t0: float,
    tf: float,
    dt: float,
) -> np.ndarray:
    """Integrate ``xdot = f(t, x)`` from ``t0`` to ``tf`` with classic RK4.

    Returns the state at ``tf``.  ``RK4.m`` also returns the whole trajectory,
    but every caller in the thesis only uses ``xout(end, :)``.

    The step count is ``round((tf - t0) / dt)``, which is what MATLAB's
    ``for t = t0:dt:tf-dt`` gives for the step sizes used (``dt/5``, ``dt/10``).
    """
    x = np.asarray(x0, dtype=float).copy()
    n = int(round((tf - t0) / dt))
    t = t0
    for _ in range(n):
        k1 = f(t, x)
        k2 = f(t + dt / 2, x + k1 * dt / 2)
        k3 = f(t + dt / 2, x + k2 * dt / 2)
        k4 = f(t + dt, x + k3 * dt)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        t += dt
    return x
