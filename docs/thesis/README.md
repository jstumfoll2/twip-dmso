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
      ch5_model.tex           DC motor + TWIP dynamics
      ch6_control.tex         LQR + command-filtered neural backstepping
      ch7_simulation.tex      discretization, noise, nonlinearities, KF baselines
      ch8_results.tex         test matrix, metrics, figure slots
      ch9_conclusions.tex
      appA_validation.tex     implementation and test guidance
      appB_errata.tex         technical changes from revision 7
    figures/                  drop figures here; see notes below

## Figures

All 18 figure slots are filled. `figures/` is written by

    uv run python scripts/thesis_rev8.py        # from the repo root

which also writes the Chapter 8 table bodies to `generated/tab_*.tex` and every
number quoted in the text to `generated/results.json`. `twip_robot.jpg`,
`fbd_wheel.png` and `fbd_pendulum.png` are copied from the original thesis
folder (`Pictures/`, `Thesis/Diagrams/`); everything else is generated. The
`\figslot` macro is still defined in main.tex for future placeholders.

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
* Chapter 8: per-case metrics; results to be regenerated.
* Chapter 5: the theta = pi convention is relabelled. Revision 7 called it the
  statically stable point; the gravity term sign in the linearization shows it
  is the unstable upright equilibrium.

## Regenerating results

Chapter 8 is populated from `scripts/thesis_rev8.py`, which runs the test
matrix of Section 8.1 on the corrected model (Section 7.2). The implementations
are `twip.observers` (DMSO rev 8, Kalman baselines), `twip.controllers` (LQR,
two-step, command-filtered backstepping) and `twip.experiments` (cases).

Key findings recorded in the text: the rev-8 DMSO beats the augmented-state
KF by ~11x on uncertainty identification without noise, but not with noise;
the command-filtered controller (Ch. 6) is unstable for all gains because it
inverts the right-half-plane zero of x1/u (Section 8.3.1 and the
"Realizability" remark in Chapter 6).
