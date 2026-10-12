# The 2020 ACC DMSO against the revision-8 DMSO

Paper: J. Stumfoll, J. Yao and S. N. Balakrishnan, "Neural Network Based Discrete Time
Modified State Observer: Stability Analysis and Case Study", 2020 American Control
Conference, pp. 2484-2489. Below, "the paper" and "ACC 2020".

The paper's observer is `twip.observers.DMSO2020`. Every number below comes from
`uv run python scripts/compare_dmso2020.py` and is stored in [`results.json`](results.json).

## Summary

- **Same observer family, one real difference.** The paper fixes two of revision 7's
  problems the same way revision 8 does: the basis is evaluated on the measurement, and the
  weights are updated with the next error `e_{k+1}`. It also adds a leakage term. Read with
  the induced 2-norm, its update law is exactly revision 8's sigma-modification with
  `Gamma_8 = alpha` and `kappa = Gamma / alpha` (the code agrees to 2e-16). What really
  differs is the error injection: the paper keeps `K = I`, as revision 7 and the v9
  firmware did, while revision 8 uses `K = F - 0.5 I`.
- **On the thesis's observer cases** (corrected plant, Chapter 8), the paper's observer
  predicts the state 10-17% better without noise (O2) but identifies the uncertainty
  about 2x worse. With sensor noise (O3, O4) it is worse on both, and its velocity error
  is 2.6x revision 8's because `K = I` passes the encoder quantization noise straight
  through. Most of the uncertainty gap comes from `K = I`, the rest from the leakage.
- **On the paper's own case study the leakage is the right call.** There the uncertainty
  dies out with the state, and the leak lets the weights forget a huge initial transient.
  Revision 8's DMSO, which has no leak, does worse on that run.
- **The stability proof drops a term, and condition (10), `0 < Gamma < 2`, is not
  sufficient.** With all three of the paper's conditions satisfied, the weights diverge
  for `Gamma >= 1.98`. Gains anyone would use (`Gamma` much smaller than `alpha`) are not
  affected.
- **On the robot's two v9 logs neither observer wins outright.** The paper's predicts the
  next measurement better, and revision 8's tilt agrees with the gyro 1.5 times better.
  This is the same trade as in simulation, and the logs have no truth to settle it.
- **The case study runs on revision 7's model** (motor-torque sign error included), starts
  at 10 rad of tilt with up to 540 V of unsaturated input, and its figures match a 1 ms
  step rather than the stated 10 ms (inferred from the figures). Nothing in the paper runs
  on hardware.

## 1. The observers side by side

| | Revision 7 | ACC 2020 | Revision 8 |
|---|---|---|---|
| Basis input | estimate `xhat_k` | measurement `y_k` | measurement `y_k` |
| Basis | `[1, tanh(xhat)]` | `[1, tanh(y)]`, "normalized" (how is not stated) | `[1, tanh(y / s)]`, `s = (1, 1, 0.2, 1)` |
| Error in the update | differs by script | `e_{k+1}` | `e_{k+1}` |
| Leakage | none | `-Gamma \|\|I - alpha phi phi^T\|\| W_k` | none (sigma-mod is an option) |
| Error injection | `K = I` | `K = I` | `K = F - 0.5 I`, so `A = 0.5 I` |
| Learning rate | 0.1 | `alpha = 0.01` | `Gamma = 0.01`, a tenth of the bound |
| Leakage gain | | not reported | `kappa = 0.1` on the sigma-mod row |
| Stability result | invalid (Appendix B.1) | UUB of `e` and `W~` under (9)-(11) | mean-square bound; UUB with bounded `W~` |

**The update law is revision 8's sigma-modification.** `I - alpha phi phi^T` has
eigenvalues 1 and `1 - alpha ||phi||^2`, so its 2-norm is 1 whenever
`alpha ||phi||^2 <= 2`, and (6) becomes
`W_{k+1} = (1 - Gamma) W_k + alpha phi_k e_{k+1}^T`. That is revision 8's eq. (sigmod)
with `Gamma_8 = alpha` and `kappa = Gamma / alpha`. The paper never says which matrix norm
it means. With the Frobenius norm the factor is about 2.2 for five basis functions, which
only rescales `Gamma`. `DMSO2020` takes `leak_norm="2"` (the default) or `"fro"`. The
2-norm is the reading under which (10) at least keeps the leak itself contracting.

**Choices the paper leaves open.** `Gamma` is not reported. The default `gamma=1e-3`
reproduces the paper's Figure 2 (section 3), and it is also the leakage per step of
revision 8's sigma-mod row. The basis normalization is not described, so the default
basis is unscaled. Section II has no selection matrix (`f` in R^n), but Section IV assigns
uncertainty to the velocity and tilt-rate rows only, so the default `Bm` is revision 8's.
`Bm=np.eye(4)` gives the Section II form.

## 2. The stability proof

What the paper does better than revision 7: it keeps `||I - alpha phi phi^T||` instead of
the invalid `(1 - Gamma ||phi||^2)` bound (Appendix B.1.1), it weights the Lyapunov
function by `1/alpha` directly (B.1.2), and its leakage bounds the weights without
persistent excitation. Revision 8 only sketches that last part for sigma-modification
(Section 3.6.2). Once repaired, explicit bounds `b_e`, `b_W` for the leaky law would be a
useful addition to Chapter 3.

Where it fails:

1. **A term is lost between (30) and (31).** The step bounds
   `-tr(W~^T M W~)` by `-||M|| ||W~||^2`, `M = I - alpha phi phi^T`. That needs the
   smallest eigenvalue of `M`, not its norm. Exactly, `tr(W~^T M W~) = ||W~||^2 - alpha ||ebar||^2`,
   so the first difference carries an extra `+2 Gamma ||ebar||^2`. The coefficient of
   `ebar^T ebar` in (32) should be `-(1 - alpha phi^T phi - 2 Gamma)`, not
   `-(1 - alpha phi^T phi)`. The closed form is in `first_difference_check` in the
   script. It matches the difference computed directly from (6) and (16) to 1e-13 on
   2000 random draws.
2. **Condition (10) is not sufficient.** With the regressor frozen, the weight error is
   multiplied each step by `1 - Gamma ||M|| - alpha ||phi||^2` along `phi`. The weights
   therefore need `Gamma ||M|| + alpha ||phi||^2 < 2`, which
   `DMSO2020Conditions.weights_contractive` checks. Conditions (9)-(11) allow `Gamma` up to
   2 whatever `alpha ||phi||^2` is. On the paper's own case at 1 ms, (9)-(11) hold for
   every `Gamma` tried up to 1.99. Yet the weights diverge for `Gamma = 1.98` and `1.99`
   (2-norm), and for every `Gamma >= 1` with the Frobenius norm (`gamma` step;
   `tests/test_dmso2020.py` has a minimal case). Both problems sit at large `Gamma`. At
   `Gamma` much smaller than `alpha`, the lost term is negligible.
3. **Smaller slips.** (16) writes `A = F - K_mso`. (33) flips the sign of the
   `(lambda_min(Q) - ...)||e||^2` term relative to (32). (33)-(35) use the matrix
   `I - alpha phi phi^T` where the scalar `1 - alpha phi^T phi` is meant. `eta`, `kappa`
   and `zeta` depend on `phi(x_k)`, so the bounds (37)-(39) are not constants until
   `phi(x_k)` is replaced by `phi_N`. (11) uses `alpha sigma_max(A)^2 eta` while (32) uses
   `alpha phi_N^2 sigma_max(A)^2`. With `Q = I - A^T A`, (11) needs `sigma_max(A) < 1`,
   the same requirement as revision 8's (C2).

## 3. The paper's case study, rebuilt

Rebuilt from Table 1 and Section IV (`paper` step):

- **Model.** Eq. (41) is the revision-7 linear model (`twip.legacy.dynamics.linear_model`),
  equal to machine precision. Table 1's "DMSO values" are the revision-7 parameters
  (`params.THESIS`). Its "true values" are revision 7's uncertainty case with
  `km x 2.5` (`THESIS.with_thesis_uncertainty(l_scale=1.0)`). The case study therefore
  carries the revision-7 model errors that revision 8 corrected: the motor reaction-torque
  sign (PR #1, thesis Section 5.3), `ke` 85 times too small, back-EMF on the absolute wheel
  rate, and `Ip` about the axle. Revision 8 reduced the `x 2.5` perturbation because it
  makes `epsilon_N`, and with it the observer bound, vacuous (Section 8.1).
- **Discretization.** Eq. (44), `G = F [I - exp(-A dt)] A^{-1} B`, needs `A` invertible.
  `A` in (41) has rank 3 because its first column is zero. The zero-order-hold integral
  (`twip.control.c2d_zoh`) is what has to be used.
- **Operating range.** The run starts at 10 rad of tilt (573 deg), and the LQR asks for up
  to 540 V. Position reaches -163 m and velocity -130 m/s. On a linear model that is legal
  arithmetic, but none of it is near the robot's range. The truth is the same discrete
  linear model with other parameters, so the case study tests neither the linearization
  nor the discretization error that the conclusion lists.
- **Time step (inferred).** The figures match `dt = 1 ms` with the time axis showing
  step x 0.01, not the stated 10 ms. At 1 ms the rebuilt run gives:
  - `f(0) = (-0.494, -0.289)`, against axes that bottom out at -0.5 and -0.3;
  - a position minimum of -163 m at step 2575 (Fig. 1: about -160 m near "25 s");
  - a velocity minimum of -130 m/s at step 369 (near "5 s");
  - a tilt minimum of -1.5 rad at step 1002 (near "10 s");
  - a peak of `f_1` of 0.036 at step 816 (Fig. 2: 0.035 near "8 s").

  At 10 ms each of these is ten times larger or ten times sooner. This is inferred from
  the figures; the simulation code would settle it.
  ![Figure 2 rebuilt at 1 ms and 10 ms](paper_fig2.png)
- **Fig. 1 does not show the network working.** With `K = I` the prediction is the
  measurement plus one step of model, `e_{k+1} = (F - I) e_k + (f - fhat)`, and
  `sigma_max(F - I) = 0.04` at 1 ms. With the network switched off (`alpha = 0`), the
  largest prediction error is 0.49 m/s and 0.29 rad/s, on velocity and tilt-rate ranges
  of 162 m/s and 30 rad/s. The plot would look the same.
- **Fig. 2 is reproducible, but only at the right `Gamma` and step.** At 1 ms with
  `Gamma = 1e-3`, the estimate tracks `f` to 1.5% RMS after the first 500 steps. It is
  sensitive to the unreported `Gamma`: 22% at `Gamma = 0`, 20% at 1e-2, 71% at 1e-1. At
  the stated 10 ms, the same gains leave 82% and 95% of `f` unexplained; the best value
  tried leaves 23-29%.
- **Leakage is what makes Fig. 2 work.** The uncertainty, `f = (dF - dG K_LQR) x`, dies out
  as the state regulates, and the weights learned during the saturated (`tanh`)
  transient have to be forgotten. Revision 8's DMSO (no leak) on the same run leaves
  15-17% at 1 ms, and an error 2.6-3.0 times the RMS of `f` at 10 ms.
- **Case II (noise).** With `K = I` the estimate error equals the measurement noise: 1.00x
  in tilt and 1.02x in tilt rate. (This uses white noise at Table 3's `sigma_r^2`, without
  the walking bias.) Revision 8's `A = 0.5 I` gives 0.58x and 0.61x. In Fig. 4 the
  "true system" trace is as noisy as the estimate, so it appears to be the measurement. If
  so, the figure shows the estimate following the noise (inferred from the figure).
- **Table 3.** The variances are listed in rad and rad/s. Two values differ from the thesis:
  the tilt `sigma_r^2` is 4.3097e-6 against 4.0397e-6 in revision 7, and the tilt
  `sigma_br^2` is 1.8929e-2 against 1.8929e-7.

## 4. Head to head on the thesis's observer cases

These are the Chapter 8 cases on the corrected plant (`matrix` step), scored after the
first 5 s. The prediction column is the one-step RMS error in velocity (mm/s) / tilt
(mrad) / tilt rate (mrad/s). The `fhat` column is the RMS uncertainty error (x1e-3) in
velocity / tilt rate. The RMS of the true `f` is 1.99 / 12.4.

| Estimator | O2 prediction | O2 `fhat` error | O3 prediction | O3 `fhat` error |
|---|---|---|---|---|
| Augmented-state KF | 0.70 / 0.09 / 11.3 | 0.69 / 11.3 | 1.82 / 12.8 / 1.04 | 0.59 / 7.28 |
| DMSO rev. 8 | 0.88 / 0.13 / 5.62 | 0.44 / 2.81 | 2.59 / 19.4 / 5.07 | 0.94 / 11.4 |
| DMSO rev. 8 + sigma-mod | 1.04 / 0.13 / 6.50 | 0.52 / 3.25 | 2.82 / 19.4 / 6.62 | 1.12 / 12.6 |
| **DMSO ACC 2020** | 0.80 / 0.11 / 4.99 | 0.91 / 5.67 | 6.85 / 19.4 / 7.00 | 1.44 / 13.4 |
| ACC 2020, `Gamma = 0` | 0.63 / 0.10 / 3.99 | 0.72 / 4.52 | 6.81 / 19.4 / 6.12 | 1.22 / 12.0 |
| ACC 2020, `K = F - 0.5 I` | 1.02 / 0.13 / 6.35 | 0.51 / 3.17 | 2.83 / 19.4 / 7.02 | 1.13 / 12.7 |
| DMSO rev. 7 (outside its proven region) | 0.10 / 0.07 / 0.69 | 0.11 / 0.75 | 6.28 / 19.4 / 4.44 | 1.22 / 11.6 |

![Case O2 uncertainty estimates](O2_uncertainty.png)

- **`K = I` is the main difference.** It trades uncertainty identification for state
  prediction: a faster correction leaves less of the error for the network to learn. In
  O2 the `fhat` error is 0.91 / 5.67 with `K = I`, against 0.51 / 3.17 with
  `K = F - 0.5 I` at the same `alpha` and `Gamma`. With noise it costs the state estimate
  too, because `K = I` hands the measurement through. The velocity error of 6.85 mm/s is
  the measurement's own quantization error (6.27 mm/s).
- **The leakage costs 16-27% here** because the uncertainty persists in a tracking run, as
  thesis Section 8.2 says of sigma-modification. The basis normalization changes the
  `fhat` error by under 3% (the `K = F - 0.5 I` row against the sigma-mod row, which is
  the same law with revision 8's basis).
- **In O3 the paper's tilt-rate uncertainty error** (13.4e-3) exceeds the RMS of the
  uncertainty itself (12.4e-3), so it does worse than estimating zero. Revision 8 is at
  92%.
- **O4 matches O3 within a few percent, and O1 is near zero for every observer**
  (`results.json`). The paper's conditions (9)-(11) hold in every case with `K = I`
  (`sigma_max(F - I) = 0.92` on the corrected model).

## 5. Hardware

The paper has no hardware results, and its remark that the DMSO "can be used in embedded
systems easily" is not tested there. The robot did run a close relative in 2015: the v9
firmware DMSO had `K = I`, `gamma = 0.05`, no leakage and the basis on the estimate.

The `hardware` step replays the two v9 balancing logs (`implementationtest4.txt`, 12.6 s,
and `implementationtest5.txt`, 21.2 s) through both observers on the corrected model. It
uses the frame of the thesis replays (logged position mirrored) and scores from four
samples after switch-on (sample 310 of test 5, as in the thesis). There is no truth on
hardware. Each observer is scored against the next measurement, and its tilt against the
gyro, `rms(d tilt/dt - gyro)`.

| Test 4 / test 5 | One-step residual: velocity (mm/s) | tilt (mrad) | tilt rate (mrad/s) | Tilt vs gyro (deg/s) | RMS `fhat`, tilt rate (x1e-3) |
|---|---|---|---|---|---|
| Measurement (complementary tilt) | | | | 18.4 / 9.9 | |
| DMSO rev. 8 | 20.9 / 15.9 | 3.47 / 2.25 | 229 / 147 | 9.5 / 5.9 | 21 / 18 |
| DMSO ACC 2020 | 22.3 / 12.0 | 3.07 / 1.85 | 218 / 113 | 14.2 / 8.3 | 44 / 26 |
| On-board v9 DMSO (logged) | 36.2 / 23.5 | 3.68 / 2.25 | 233 / 110 | 18.1 / 10.6 | |

The logs show the same trade as the simulations, and neither observer wins outright:

- **The paper's observer predicts the next measurement better.** Its tilt and tilt-rate
  residuals are smaller in both logs, and its velocity residual is smaller in test 5.
- **Revision 8's tilt is smoother.** Its tilt agrees with the gyro 1.5 times better in both
  logs, and both observers are smoother than the measurement.

That is what `K = I` does: it follows the measurement, noise included. On a robot whose
tilt measurement carries the accelerometer's acceleration error (thesis Section 8.4), the
smoother estimate is the more useful one. That is a judgment, though, since the logs have
no truth to settle it.

The paper's tilt-rate uncertainty estimate is 1.4 to 2 times larger. Without truth, the
logs cannot say how much of it is model error and how much is noise. In simulation (O3)
the larger estimate was mostly noise.

The revision-8 numbers for test 5 match the thesis: 5.9 deg/s against the gyro, and an
`fhat` RMS of 2.5e-3 and 1.8e-2 (Section 8.4).

## 6. What was added to the code

- `twip.observers.DMSO2020` and `DMSO2020Conditions`: the paper's eqs. (3), (5) and (6).
  The defaults are `K = I`, `alpha = 0.01`, `gamma = 1e-3`, an unscaled basis, `Bm` on
  rows 2 and 4, and the 2-norm. `conditions()` reports (9)-(11) and the contraction
  condition.
- `twip.experiments.make_estimators(..., acc2020=True)` adds a `"DMSO (ACC 2020)"` row.
  The default is unchanged, so the thesis tables do not change.
- `scripts/compare_dmso2020.py` produces everything in this document.
- `tests/test_dmso2020.py` covers the paper defaults, the norm, the equivalence with
  revision 8's sigma-mod, identification of a constant uncertainty, the conditions and
  the condition (10) counterexample.

## 7. Recommendation

Keep revision 8's DMSO (`K = F - sigma_bar I`) as the default. It identifies the
uncertainty better in the cases the thesis cares about and filters noise that `K = I`
passes through.

From the paper, the leakage is worth using when the network should forget a large
transient. It is the same law as revision 8's sigma-mod, so nothing new is needed in the
code to get it. The explicit weight bound is worth having in Chapter 3, once the lost
`2 Gamma ||ebar||^2` term is restored and (10) is replaced by a condition that keeps the
weight error contracting. If the thesis cites the paper, it should note that the paper's
case study uses the revision-7 model.
