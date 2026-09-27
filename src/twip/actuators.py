"""Actuator nonlinearities: saturation, deadzone and backlash."""

from __future__ import annotations

import numpy as np


def saturate(u: float, limit: float) -> float:
    """``if abs(u) > sat, u = sign(u)*sat``."""
    return float(np.clip(u, -limit, limit))


def deadzone(u: float, width: float) -> float:
    """``if abs(u) < dz, u = 0``: a hard deadzone with a jump."""
    return 0.0 if abs(u) < width else u


class Backlash:
    """Discrete backlash (hysteresis) element with slope ``m``.

    This is the form used for the tilt-angle backlash in ``ExtraControl_v5.m``
    (the ``ubs``/``ubs2`` variables).  The output only moves once the input
    has crossed the dead band ``[d_minus, d_plus]`` around it.
    """

    def __init__(self, y0: float, m: float = 1.0, d_plus: float = 0.0, d_minus: float = 0.0):
        self.y = y0
        self.m = m
        self.d_plus = d_plus
        self.d_minus = d_minus

    def step(self, v_new: float, v_prev: float) -> float:
        m, dp, dm = self.m, self.d_plus, self.d_minus
        if (v_new - v_prev > 0) and (self.y <= m * (v_prev - dp)):
            self.y = m * (v_new - dp)
        elif (v_new - v_prev < 0) and (self.y >= m * (v_prev - dm)):
            self.y = m * (v_new - dm)
        return self.y
