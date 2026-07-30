# Changelog

## 1.0.2 - 2026-07-30

- Added `scripts/validation_experiments.py` and the `models/validation_experiments.json`
  artifact: a seven-step capacity sweep and a three-way split comparison that rule
  out model capacity and data volume as causes of the weak signal.
- Quantified the cost of a wrong split: a naive random split scores AUC 0.581 instead
  of 0.519 on the same data, making the apparent signal 4.2x larger.
- Corrected the learning-curve interpretation across all artifacts. The curve shows
  high variance (gap 0.112), not high bias; the previous "no variance lever" claim
  contradicted the app's own diagnosis in `src/diagnostics.py`.
- Made the learning-curve remedy in `src/diagnostics.py` data-aware: it no longer
  recommends "more data" when the validation curve is flat near chance level.
- Reframed the negative result from a caveat into the central finding in the report,
  the handout, notebooks 04 and 05 and the presentation.
- Restructured the presentation: live demo moved to the middle (18:10), two dedicated
  question windows, evidence slides moved to the appendix, new slide "Was Methodik
  wert ist", full speaker script in the notes of all 37 slides.
- Added a direct link to the application plus local start instructions to the report.
- Switched all report tables to German decimal notation and completed the table
  numbering.
- Made the version a single source of truth in `src/config.py`; the project page,
  both generator scripts and the CHANGELOG are now guarded by a test that fails on
  drift. Note: `models/diagnostics.json` still records `app_version` 1.0.0, which is
  correct provenance - the trained model itself is unchanged since 1.0.0.
- Added a prominent "Zugang zur Anwendung" section with the repository link and the
  three start commands to the README, mirroring section 9 of the report.
- Corrected the README: the Random Forest pipeline does not include a StandardScaler
  (trees are scale-invariant; the scaler applies to Logistic Regression and Linear
  SVM only), and the actual result was never stated - it now leads with the metrics
  table and the falsification of H1.
- Cleaned up the repository: removed the superseded 34-slide presentation and the
  separate speech script (both replaced by the 37-slide deck with speaker notes),
  and deleted the five branches that were fully merged into `main`.

## 1.0.1 - 2026-07-26

- Added an exact five-page A4 report with reproduced out-of-time metrics.
- Documented `StandardScaler` and preprocessing leakage protection explicitly.
- Limited scaling to Logistic Regression and Linear SVM; tree models remain unscaled.
- Reworked and successfully executed all eight QUA³CK notebooks.
- Replaced the obsolete AUC 0.588 claim with the reproduced AUC 0.519 result.
- Preserved the previous handout and notebooks in local `Backup` folders.

## 1.0.0 - 2026-07-26

- Replaced the random stratified split with a purged out-of-time holdout.
- Added four expanding walk-forward validation folds.
- Added a historical benchmark: Dummy, Logistic Regression, Decision Tree,
  Linear SVM and Random Forest.
- Added the interactive Lernstudio and arsnova.eu quiz export.
- Added a model card, lecture-alignment matrix and GitHub Actions checks.
- Made model and diagnostics artifacts part of the deployable repository.
- Preserved the complete pre-1.0 state in commit `834ff98` on branch
  `codex/backup-status-quo-2026-07-26`.
