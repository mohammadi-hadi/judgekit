# Changelog

## 0.1.0 — 2026-08-05

First release.

- Agreement metrics between a judge and a reference: accuracy, Cohen's kappa,
  Krippendorff's alpha, with bootstrap confidence intervals.
- Bias probes: position, verbosity, self-preference.
- Calibration: reliability curve, expected calibration error.
- Consistency across repeated runs of the same judge.
- Markdown report with figures, via the `judgekit report` command.
- Built-in demo (`judgekit demo`) that audits six synthetic judges with
  implanted defects and recovers each one.
- Fully typed (`py.typed`), Python 3.11+.
