"""Matplotlib versions of the thesis figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .sim_dmso import DMSOSimResult

R2D = 180 / np.pi


def dmso_uncertainty(r: DMSOSimResult, start: int = 0) -> plt.Figure:
    """``main_bala_discrete.m`` figure 14: actual vs. estimated uncertainty and the log error."""
    N = r.fhat.shape[1]
    t = r.time[start:N]
    f1, f2 = r.fhat[0, start:N], r.fhat[1, start:N]
    u2, u4 = r.uncertainty[1, start:N], r.uncertainty[3, start:N]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(t, u2, label="Actual")
    ax.plot(t, f1, label="Estimated")
    ax.set(title="Acceleration Uncertainty", xlabel="Time (sec)", ylabel="Acceleration (m/s)")
    ax.legend()
    ax.grid(True)
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t, u4, t, f2)
    ax.set(title="Tilt Acceleration Uncertainty", xlabel="Time (sec)", ylabel="Angular Acceleration (deg/s)")
    ax.grid(True)
    ax = fig.add_subplot(2, 1, 2)
    ax.semilogy(t, np.abs(f1 - u2), label="Acceleration")
    ax.semilogy(t, np.abs(f2 - u4), label="Tilt Acceleration")
    ax.set(title="Uncertainty Error", xlabel="Time (sec)", ylabel="Uncertainty Error")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def dmso_states(r: DMSOSimResult, start: int = 0, errors: bool = False, log: bool = False) -> plt.Figure:
    """Figures 15 (states), 16 (errors) and 17 (``|error|``, log scale).

    Note: the MATLAB code offsets the Kalman estimate by one sample in
    figure 15 (``kalman(:, k+1)`` against ``xtrue(:, k)``) but not in 16/17.
    This port plots every estimate against truth at the same index.
    """
    N = r.fhat.shape[1]
    t = r.time[start:N]
    panels = [
        (2, R2D, "Tilt Angle", "Angle (deg)"),
        (3, R2D, "Tilt Rate", "Tilt Rate (deg/s)"),
        (0, 1.0, "Position", "Distance (m)"),
        (1, 1.0, "Velocity", "Velocity (m/s)"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (k, s, title, ylabel) in zip(axs.flat, panels):
        truth, dm, kf = r.xtrue[k, start:N] * s, r.xhat[k, start:N] * s, r.kalman[k, start:N] * s
        if not errors:
            ax.plot(t, truth, label="Simulation Truth")
            ax.plot(t, dm, label="DMSO")
            ax.plot(t, kf, label="Kalman")
        else:
            plot = ax.semilogy if log else ax.plot
            plot(t, np.abs(truth - dm) if log else truth - dm, label="DMSO")
            plot(t, np.abs(truth - kf) if log else truth - kf, label="Kalman")
            title += " Error"
        ax.set(title=title, xlabel="Time (sec)", ylabel=ylabel)
        ax.grid(True)
    axs[0, 0].legend()
    fig.tight_layout()
    return fig


def control(r: DMSOSimResult, start: int = 0) -> plt.Figure:
    N = r.fhat.shape[1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r.time[start:N], r.u_unshaped[start:N], label="Before backlash/deadzone")
    ax.plot(r.time[start:N], r.u[start:N], "--", label="Applied")
    ax.set(title="Control", xlabel="Time (sec)", ylabel="Voltage (V)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def extra_control_states(r) -> plt.Figure:
    """``ExtraControl_v5.m`` figure 1: nominal, LQR-only, LQR + extra control, desired."""
    t = r.time
    panels = [
        (2, R2D, "Tilt Angle", "Angle (deg)"),
        (3, R2D, "Tilt Rate", "Tilt Rate (deg/s)"),
        (0, 1.0, "Position", "Distance (m)"),
        (1, 1.0, "Velocity", "Velocity (m/s)"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (k, s, title, ylabel) in zip(axs.flat, panels):
        nom = r.xnom[k] * s + (180 if k == 2 else 0)  # xnom is upright-relative
        ax.plot(t, nom, label="Nominal System")
        ax.plot(t, r.xtrue[k] * s, label="Perturbed System without ue")
        ax.plot(t, r.xextra[k] * s, label="Perturbed System with ue")
        ax.plot(t, r.x_des[k] * s, label="Desired")
        ax.set(title=title, xlabel="Time (sec)", ylabel=ylabel)
        ax.grid(True)
    axs[0, 1].legend()
    fig.tight_layout()
    return fig


def extra_control_inputs(r) -> plt.Figure:
    """Figure 2: optimal, nominal, extra and total control."""
    t = r.time[:-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    for v, lab in ((r.u_opt, "Optimal Control"), (r.u_nom, "Nominal Control"), (r.u_e, "Extra Control"), (r.u, "Total Control")):
        ax.plot(t, v, label=lab)
    ax.set(title="Control", xlabel="Time (sec)", ylabel="Voltage (V)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def extra_control_uncertainty(r) -> plt.Figure:
    """Figure 3: "actual" (as defined in the script) vs. NN-estimated F1, F2."""
    t = r.time[:-1]
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for k, ax in enumerate(axs):
        ax.plot(t, r.Ftrue[k], label="Actual")
        ax.plot(t, r.Fhat[k], label="Estimated")
        ax.set(title=f"F{k + 1}")
        ax.grid(True)
    axs[0].legend()
    axs[1].set_xlabel("Time (sec)")
    fig.tight_layout()
    return fig
