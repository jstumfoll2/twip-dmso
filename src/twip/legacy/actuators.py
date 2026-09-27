"""Actuator code reproduced exactly from the thesis scripts."""

from __future__ import annotations


def backlash_legacy(u_new: float, u_prev: float, m: float, d_plus: float, d_minus: float) -> float:
    """Literal port of the backlash block in ``Noise with uncertainty/main_bala_discrete.m``.

    Warning: this does nothing for the thesis settings.  The MATLAB
    conditions compare ``u(i)`` with ``m*u(i) - m*d_plus``.  With ``m = 1`` they
    reduce to ``0 <= -d_plus`` and ``0 >= -d_minus``, which are both false when
    ``d_plus > 0 > d_minus``.  So ``u`` is returned unchanged, and the
    "No Backlash" and "Backlash" traces in that script's figure 11 are identical.
    Use :class:`Backlash` for an actual backlash element.
    """
    if (u_new - u_prev > 0) and (u_prev <= m * u_prev - m * d_plus):
        return m * u_new - m * d_plus
    if (u_new - u_prev < 0) and (u_prev >= m * u_prev - m * d_minus):
        return m * u_new - m * d_minus
    return u_new


class Backlash:
    """Discrete backlash (hysteresis) element with slope ``m``.

    The rule used for the tilt backlash in ``ExtraControl_v5.m`` (the ``ubs``
    variables).  Engagement is decided from the previous input, so it lags one
    step.  See :class:`twip.actuators.Backlash` for the corrected play operator.
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
