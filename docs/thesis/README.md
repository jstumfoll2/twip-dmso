# TWIP Thesis

LaTeX sources of *Discrete-Time Neural Network Based State Observer with
Command-Filtered Neural Network Control for a Class of Systems with Unmatched
Uncertainties*.

## Build

    make            # -> main.pdf
    make clean

Requires pdflatex and bibtex. No non-standard packages beyond
amsmath/amssymb/booktabs/siunitx/tocloft/hyperref.

Note: with MiKTeX's 2026 siunitx and a 2024 l3kernel, every `\si` fails to
compile. TeX Live 2023 is fine; otherwise update MiKTeX or load
`\usepackage{siunitx}[=v2]`.

## Layout

    main.tex                  driver, preamble, macros
    frontmatter.tex           title, abstract, acknowledgments, TOC, nomenclature
    refs.bib                  bibliography
    chapters/
      ch1_introduction.tex
      ch2_literature.tex
      ch3_dmso.tex            DMSO development and Lyapunov proof
      ch4_platform.tex        hardware
      ch5_model.tex           DC motor + TWIP dynamics, model validation against the logs (5.3)
      ch6_control.tex         LQR + command-filtered neural backstepping on the flat output
      ch7_simulation.tex      discretization, noise, nonlinearities, KF baselines
      ch8_results.tex         test matrix, metrics, results
      ch9_conclusions.tex
      appA_validation.tex     implementation and test guidance
    figures/                  generated figures and the robot photographs
    generated/                table bodies and quoted numbers written by the scripts

## Figures and tables

`figures/` and `generated/` are written by

    uv run python scripts/thesis_rev8.py        # from the repo root
    uv run python scripts/validate_plant.py     # pv_*.pdf (Section 5.3)
    uv run python scripts/validate_closed_loop.py

`thesis_rev8.py` writes the Chapter 4, 5, 7 and 8 figures, the Chapter 8 table
bodies (`generated/tab_*.tex`) and every number quoted in the text
(`generated/results.json`). Its `--only hardware` step (Section 8.4:
`hw_states`, `hw_complementary`, `hw_uncertainty` and the numbers quoted there)
needs the thesis data folder; the other steps need no data.

`validate_plant.py` writes `pv_transient`, `pv_coefficients`, `pv_startup` and
`generated/plant_validation.json` (Section 5.3) and needs the thesis data folder;
it also writes `pv_frequency.pdf`, which the text does not use.
`validate_closed_loop.py` writes `generated/plant_closedloop.json` and needs no
robot data.

`twip_robot.jpg`, `fbd_wheel.png` and `fbd_pendulum.png` are copied from the
original thesis folder (`Pictures/`, `Thesis/Diagrams/`); everything else is
generated.

## Bibliography

`refs.bib` holds 60 references, all cited in the text; ieeetr numbers them by
order of first citation.

## Implementations

The results of Chapter 8 come from `twip.observers` (DMSO, Kalman baselines),
`twip.controllers` (LQR, two-step extra control, command-filtered
backstepping) and `twip.experiments` (the cases of Section 8.1), run by
`scripts/thesis_rev8.py` on the model of Section 7.2.

`docs/twip_thesis.pdf` is the current build.
