# TWIP Thesis, Revision 8

Corrected rewrite of *Discrete-Time Neural Network Based State Observer with
Neural Network Based Control Formulation for a Class of Systems with Unmatched
Uncertainties*.

## Build

    make            # -> main.pdf
    make clean

Requires pdflatex and bibtex. No non-standard packages beyond
amsmath/amssymb/booktabs/siunitx/tocloft/hyperref.

## Layout

    main.tex                  driver, preamble, macros
    frontmatter.tex           title, abstract, acknowledgments, TOC, nomenclature
    refs.bib                  bibliography
    chapters/
      ch1_introduction.tex
      ch2_literature.tex
      ch3_dmso.tex            corrected DMSO development and Lyapunov proof
      ch4_platform.tex        hardware
      ch5_model.tex           DC motor + TWIP dynamics, model validation against the logs (5.3)
      ch6_control.tex         LQR + command-filtered neural backstepping
      ch7_simulation.tex      discretization, noise, nonlinearities, KF baselines
      ch8_results.tex         test matrix, metrics, figure slots
      ch9_conclusions.tex
      appA_validation.tex     implementation and test guidance
      appB_errata.tex         technical changes from revision 7
    figures/                  drop figures here; see notes below

## Figures

All 21 figure slots are filled. `figures/` is written by

    uv run python scripts/thesis_rev8.py        # from the repo root
    uv run python scripts/validate_plant.py     # pv_*.pdf (Section 5.3)

`thesis_rev8.py` also writes the Chapter 8 table bodies to `generated/tab_*.tex`
and every number quoted in the text to `generated/results.json`.
`pv_transient`, `pv_coefficients` and `pv_startup` come from
`validate_plant.py`. `twip_robot.jpg`, `fbd_wheel.png` and `fbd_pendulum.png`
are copied from the original thesis folder (`Pictures/`, `Thesis/Diagrams/`);
everything else is generated. The `\figslot` macro is still defined in main.tex
for future placeholders.

Note: with MiKTeX's 2026 siunitx and a 2024 l3kernel, every `\si` fails to
compile. TeX Live 2023 (which built rev 8) is fine; otherwise update MiKTeX or
load `\usepackage{siunitx}[=v2]`.

## Bibliography

`refs.bib` holds all 53 references from revision 7 in their original order, each
tagged with its revision-7 number in a comment, plus 7 additions ([54]-[60])
supporting the corrected Chapter 3 and the command-filtered Chapter 6. All 60
are cited in the text; ieeetr numbers them by order of first citation, so the
printed numbers differ from the revision-7 numbers.

## What changed from revision 7

See Appendix B. In short:

* Chapter 3: the Lyapunov first difference is evaluated exactly rather than
  bounded term by term. Conditions become sigma_max(A) < 1 plus an upper bound
  on Gamma, replacing the previous conditions which were satisfiable only on a
  measure-zero boundary and which excluded small adaptation gains.
* Chapter 6: four-step command-filtered backstepping replacing the two-step
  design. Retains the A2*z3 coupling, damps all four error coordinates, and
  makes both network targets functions of measured state alone.
* Chapter 7: three Kalman baselines instead of one; the augmented-state filter
  is the one improvement claims are stated against.
* Chapter 8: per-case metrics, regenerated on the corrected model. The
  hardware replays (Section 8.4) map the logged encoder position into the model
  frame inferred in Section 5.3 (logged position mirrored, x = -pos - r*theta),
  which is not yet confirmed in closed loop.
* Chapter 5: the theta = pi convention is relabelled. Revision 7 called it the
  statically stable point; the gravity term sign in the linearization shows it
  is the unstable upright equilibrium. The motor's reaction torque on the body
  enters the pendulum moment balance with positive sign; revision 7 had it
  negative, which flipped the back-EMF and input terms of the tilt equation.
  A new remark after the linearization checks the corrected input vector
  against the angular momentum about the wheel contact point, which the motor
  torque cannot change.
* Chapter 5, new Section 5.3: the model is tested against the robot's logs, with
  three candidates (M1 corrected, +T; M2, -T; M3, revision 7). Three tests
  decide the sign for M1: the input-free angular-momentum test on the two v9
  switch-on transients, the tilt response to the saturated -10 V at switch-on,
  and the v9 loop simulated with the gains recovered from the logs, in which no
  -T plant stays up beyond 5.3 s. Some predictions favour M2 or M3 and the input
  regression is inconclusive. M1 is not validated quantitatively: the
  accelerometer points to a larger Ip, the steady-balancing coefficients
  contradict all three models, and Ip, l, km, the present motors' deadzone and
  backlash, and the floor's rolling resistance are unmeasured. The logged encoder position is inferred to
  be mirrored relative to the model; in closed loop that frame reproduces the v9
  runs only with a rolling resistance of at least about 4% of the weight, so it
  is not yet confirmed.

## Regenerating results

Chapter 8 is populated from `scripts/thesis_rev8.py`, which runs the test
matrix of Section 8.1 on the corrected model (Section 7.2). The implementations
are `twip.observers` (DMSO rev 8, Kalman baselines), `twip.controllers` (LQR,
two-step, command-filtered backstepping) and `twip.experiments` (cases).

Key findings recorded in the text: the rev-8 DMSO beats the augmented-state
KF by 1.6x (velocity) and 4x (tilt rate) on uncertainty identification without
noise, but not with noise, and a larger adaptation gain inside the proven
region does better still; the command-filtered controller as first written
(Ch. 6) is unstable for all gains because it inverts the right-half-plane zero
of x1/u (Section 8.3.1). Section 6.7 corrects it (flat output y = x1 - b x3,
eta = x2 - b x4, attitude weight a2^2). The corrected design balances in every
run and tracks 4-22x better than LQR, at 22-137% more RMS control effort.

The hardware replays of Section 8.4 (`hw_states`, `hw_complementary`,
`hw_uncertainty`, and the DMSO/KF numbers quoted there) run in the encoder frame
of Section 5.3 and are regenerated with
`uv run python scripts/thesis_rev8.py --only hardware` with the thesis data
folder present.

Section 5.3 is populated from `scripts/validate_plant.py` (needs the thesis
data folder; writes `generated/plant_validation.json` and `figures/pv_*.pdf`)
and `scripts/validate_closed_loop.py` (needs no robot data; writes
`generated/plant_closedloop.json`).

`docs/twip_thesis_rev8.pdf` is the current build.
`docs/twip_thesis_rev8_results.pdf` is an older build that predates the
motor-torque sign correction and the model validation.
