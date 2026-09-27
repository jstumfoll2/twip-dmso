# twip-dmso

Python code for a **two-wheeled inverted pendulum (TWIP) robot** and a **discrete
modified state observer (DMSO)**, starting from a 2015 master's thesis (MATLAB +
Arduino). Uses numpy, scipy and matplotlib; MATLAB is not needed.

The package has two tiers:

- **Corrected baseline** (top-level `twip`): plant, sensor, actuator and LQR baseline
  models with the thesis's modeling errors fixed. This is the reference for new
  controller and observer designs. See [`docs/BASELINE.md`](docs/BASELINE.md).
- **Legacy** (`twip.legacy`): faithful reproductions of the thesis simulations,
  observers and extra control, quirks included, so thesis results stay reproducible.

Background on the original code and every issue found in it:
[`docs/CODE_INVENTORY.md`](docs/CODE_INVENTORY.md).

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

Corrected baseline (LQR + Kalman on the corrected plant):

```bash
uv run python scripts/run_baseline.py
```

```bash
uv run python scripts/run_baseline.py --scale ke=0.5 --tilt 2
```

Thesis revision 8 (`docs/thesis`): regenerate every Chapter 8 figure and table:

```bash
uv run python scripts/thesis_rev8.py
```

Thesis reproductions and data replays:

```bash
uv run python scripts/legacy/run_dmso_sim.py noise_with_uncertainty
```

```bash
uv run python scripts/legacy/run_extra_control.py unmodeled_dynamics
```

```bash
uv run python scripts/run_data_replays.py firmware
```

| Script | Options |
|---|---|
| `run_baseline.py` | `--estimator kalman/complementary`, `--gains lqr/hardware`, `--tilt`, `--scale PARAM=FACTOR`, `--delay`, `--no-noise` |
| `legacy/run_dmso_sim.py` | `no_noise_no_uncertainty`, `no_noise_with_uncertainty`, `noise_with_uncertainty`, `noise_with_uncertainty_perturbed` |
| `legacy/run_extra_control.py` | `as_saved`, `unmodeled_dynamics`, `parameter_uncertainty`, `deadzone`, `deadzone_and_backlash` |
| `run_data_replays.py` | `filters`, `implementation` (`--thesis-r-bug`), `firmware`, `allan` |

All scripts take `--out DIR` to save PNGs instead of opening windows.

## Layout

| Module | Contents |
|---|---|
| `twip.params` | `CORRECTED` and `THESIS` parameter sets, motor-constant derivations, `scaled()` |
| `twip.dynamics` | corrected nonlinear EOM (mass-matrix form) and linearization |
| `twip.sensors` | accelerometer kinematics, gyro, relative encoders, noise from the Allan analysis |
| `twip.actuators` | firmware voltage path (`Actuator`), saturation, deadzone, play-operator backlash |
| `twip.baseline` | `design_lqr`, `KalmanEstimator`, `ComplementaryEstimator`, `simulate`, metrics |
| `twip.observers` | thesis revision 8: corrected DMSO (Ch. 3), the three Kalman baselines (Ch. 7), revision-7 DMSO adapter |
| `twip.controllers` | thesis revision 8: LQR tracking, revision-7 two-step extra control, command-filtered backstepping (Ch. 6) |
| `twip.experiments` | the Chapter 8 test matrix (observer cases O1-O4, control cases C1-C4) |
| `twip.estimators` | linear Kalman, tilt/gyro-bias Kalman, complementary filter |
| `twip.control` | `c2d`, `lqr`, `dlqr`, `lqrd` (Control System Toolbox replacements) |
| `twip.integrators` | fixed-step RK4 (`RK4.m`) |
| `twip.analysis` | replays of logged robot data: filter comparison, hardware LQR runs, v9 firmware DMSO, Allan variance |
| `twip.data` | `.mat` and serial-log loaders |
| `twip.legacy.dynamics` / `.sensors` | the thesis EOM and sensor model |
| `twip.legacy.observers` | thesis DMSO (4-state, 2-state) and the v9 on-board DMSO |
| `twip.legacy.sim_dmso` | `main_bala_discrete.m`, one preset per thesis folder |
| `twip.legacy.extra_control` | `ExtraControl_v5.m` + `NN1v3.m` |
| `twip.legacy.plots` | the thesis figures |

## Legacy quirks and their switches

The legacy tier reproduces what the MATLAB and firmware actually did. Where a quirk
lives outside `twip.legacy` (the data replays), the default is now the corrected
behaviour and a switch reproduces the thesis:

| Quirk | Default | Switch |
|---|---|---|
| Backlash in the DMSO sim is a no-op | reproduced in `legacy.sim_dmso` | `actuators.Backlash` is the correct form |
| v4 firmware MSO `F[1][0]` bug | reproduced in `replay_filters` (it's what the robot ran) | `firmware_bug=False` |
| In-place Kalman covariance update | textbook | `AngleBiasKalman(literal=True)` (used by `replay_filters`) |
| Scalar `R = .1` overwrite in the hardware replay | intended R | `reproduce_R_bug=True` |
| DMSO weight-update innovation index | per preset | `DMSOSimConfig.weight_innovation` |
| Extra control `ke0 = 1` (diverges) | `as_saved` preset only | figure presets use `ke0 = 0` |
| NN2 trained on `e4` instead of `ebar4` | reproduced | `ExtraControlConfig.nn2_error` |

Unavoidable differences from MATLAB:
- **Random numbers**: MATLAB's `randn`/`rand` streams can't be reproduced. Noisy runs and the
  extra control's random NN input layer are seeded (`seed=0`), so they match statistically, not sample for sample.
- `sind(theta*180/pi)` becomes `sin(theta)` (only the rounding differs).
- `lqrd` is reimplemented (Van Loan cost discretization + DARE with a cross term).

## Verification

Legacy code is checked against **saved outputs of the original code**:

| Check | Reference | Agreement |
|---|---|---|
| Complementary + tilt/bias Kalman filters | robot log `filteringtest12.mat` (v4 firmware) | < 0.02 deg (log printed to 3 dp) |
| 2-state DMSO with the F bug | same log (`pitchmso`, `fhat1/2`) | < 0.005 deg |
| v9 on-board 4-state DMSO | robot logs `implementationtest4/5.txt` | < 0.002 (print precision) |
| Allan deviation | MATLAB `avar` stored in `allandata.mat` | 2e-11 relative |
| Extra control (nominal, LQR-only, extra) | saved figures in `Extra control/Deadzone*`, `Unmodeled Dynamics` | peak values within a few % |
| DMSO simulation | saved thesis figure (`Noise with uncertainty/figure2.jpg`) | qualitative (the figure predates the final gains) |

The corrected baseline is checked against physics:
- The corrected EOM reduce to the thesis EOM exactly when the back-EMF simplification is
  switched back on.
- The linear model is the Jacobian of the nonlinear one.
- The motors are dissipative, and the accelerometer and encoder models behave as expected.

It is **not** validated against hardware data: the closed-loop logs cannot discriminate
between parameter sets. See [`docs/BASELINE.md`](docs/BASELINE.md) for the evidence and
suggested bench tests.

## Not ported

- `animation.m` (3-D robot animation).
- `Tests/LQR Tests` (~25 near-duplicate tuning scripts). `replay_implementation` covers the final version.
- DMSO gain tuning (`maintuning.m`, `fminsearchbnd`); `scipy.optimize.minimize(..., bounds=...)` would replace it.
- Motor tests / motor constant identification (`Tests/Motor Tests`); they used the earlier 6 V motors.
- Sliding-mode control (root `slidingmode*.m`, firmware v5-v9). It was explored but is not a thesis result.
- The Simulink models in `Model/` (earliest iteration, superseded).
