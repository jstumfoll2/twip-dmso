# twip-dmso

A Python port of the MATLAB and Arduino code from a master's thesis on a **two-wheeled
inverted pendulum (TWIP) robot** and a **discrete modified state observer (DMSO)**.
The DMSO estimates the robot's state and its model uncertainty. The thesis compares it
with Kalman and complementary filters, and adds neural-network "extra control" on top of
LQR. The port uses numpy, scipy and matplotlib; MATLAB is not needed.

- What the original folder contains, and the bugs found in it: [`docs/CODE_INVENTORY.md`](docs/CODE_INVENTORY.md)

## Setup

```bash
uv sync
```

```bash
uv run pytest
```

Tests that need the thesis data files (robot logs, `.mat` files) look for the original
thesis folder one level above this repo. Set `TWIP_THESIS_ROOT` to point elsewhere.
Without the data, those tests are skipped and the rest still run.

## Running

```bash
uv run python scripts/run_dmso_sim.py noise_with_uncertainty
```

```bash
uv run python scripts/run_extra_control.py unmodeled_dynamics
```

```bash
uv run python scripts/run_data_replays.py firmware
```

| Script | Presets / commands |
|---|---|
| `run_dmso_sim.py` | `no_noise_no_uncertainty`, `no_noise_with_uncertainty`, `noise_with_uncertainty`, `noise_with_uncertainty_perturbed` |
| `run_extra_control.py` | `as_saved`, `unmodeled_dynamics`, `parameter_uncertainty`, `deadzone`, `deadzone_and_backlash` |
| `run_data_replays.py` | `filters`, `implementation`, `firmware`, `allan` |

All scripts take `--out DIR` to save PNGs instead of opening windows.

## Layout

| Module | Ports |
|---|---|
| `twip.params` | the parameter block copy-pasted into every script; `with_thesis_uncertainty()` |
| `twip.dynamics` | `twipnonlinear.m` (both angle conventions, optional unmodeled terms), linearized A/B |
| `twip.control` | `c2d`, `lqr`, `dlqr`, `lqrd` (Control System Toolbox replacements) |
| `twip.integrators` | `RK4.m` |
| `twip.estimators` | 4-state DMSO, 2-state DMSO (`DMSO.m` / v4 `MSO.ino`), v9 on-board DMSO (`FirmwareDMSO`), linear Kalman, tilt/bias Kalman, complementary filter |
| `twip.actuators` | saturation, deadzone, backlash |
| `twip.sensors` | encoder quantization, white noise + Gauss-Markov bias IMU model |
| `twip.sim_dmso` | `main_bala_discrete.m`, one preset per thesis folder |
| `twip.extra_control` | `ExtraControl_v5.m` + `NN1v3.m` |
| `twip.analysis` | `filteringtest.m`, `lqrkalmantest1v2.m`, v9 firmware replay, `allan.m` / `allanprocess.m` |
| `twip.data` | `.mat` and serial-log loaders (replaces the `*import*.m` scripts) |
| `twip.plots` | the thesis figures |

## Translation policy

The port is **faithful by default**. It reproduces what the MATLAB and firmware
actually did, quirks included, so results can be compared with the thesis. Each
known quirk is documented where it lives and has a switch:

| Quirk | Default | Switch |
|---|---|---|
| Backlash in the DMSO sim is a no-op | reproduced (`backlash_legacy`) | use `actuators.Backlash` |
| v4 firmware MSO `F[1][0]` bug | reproduced in `replay_filters` | `firmware_bug=False` |
| In-place Kalman covariance update | reproduced | `AngleBiasKalman(literal=False)` |
| Scalar `R = .1` overwrite in the hardware replay | reproduced | `reproduce_R_bug=False` |
| DMSO weight-update innovation index | per preset | `DMSOSimConfig.weight_innovation` |
| Extra control `ke0 = 1` (diverges) | `as_saved` preset only | figure presets use `ke0 = 0` |
| NN2 trained on `e4` instead of `ebar4` | reproduced | `ExtraControlConfig.nn2_error` |

Unavoidable differences from MATLAB:
- **Random numbers**: MATLAB's `randn`/`rand` streams can't be reproduced. Noisy runs and the
  extra control's random NN input layer are seeded (`seed=0`), so they match statistically, not sample for sample.
- `sind(theta*180/pi)` becomes `sin(theta)` (only the rounding differs).
- `lqrd` is reimplemented (Van Loan cost discretization + DARE with a cross term).

## Verification

With no MATLAB available, the port is checked against **saved outputs of the original code**:

| Check | Reference | Agreement |
|---|---|---|
| Complementary + tilt/bias Kalman filters | robot log `filteringtest12.mat` (v4 firmware) | < 0.02 deg (log printed to 3 dp) |
| 2-state DMSO with the F bug | same log (`pitchmso`, `fhat1/2`) | < 0.005 deg |
| v9 on-board 4-state DMSO | robot logs `implementationtest4/5.txt` (`pitchm`, `xhat`, `xhatdot`, `gyhat`) | < 0.002 (print precision) |
| Allan deviation | MATLAB `avar` stored in `allandata.mat` | 2e-11 relative |
| Linear model | Jacobian of the nonlinear EOM | 1e-6 relative |
| Extra control (nominal, LQR-only, extra) | saved figures in `Extra control/Deadzone*`, `Unmodeled Dynamics` | peak values within a few % |
| DMSO simulation | saved thesis figure (`Noise with uncertainty/figure2.jpg`) | qualitative (the figure predates the final gains) |

The extra-control and DMSO-figure checks are values read off saved JPEGs, so they are approximate.

## Not ported

- `animation.m` (3-D robot animation).
- `Tests/LQR Tests` (~25 near-duplicate tuning scripts). `replay_implementation` covers the final version.
- DMSO gain tuning (`maintuning.m`, `fminsearchbnd`); `scipy.optimize.minimize(..., bounds=...)` would replace it.
- Motor tests / motor constant identification (`Tests/Motor Tests`).
- Sliding-mode control (root `slidingmode*.m`, firmware v5-v9). It was explored but is not a thesis result.
- The Simulink models in `Model/` (earliest iteration, superseded).
