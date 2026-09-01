# Cross-validated results on the public training set

These are **out-of-fold** numbers: 5-fold cross-validation over the 75 public patients, each
patient scored by the model of the fold in which it was held out. This is the honest estimate of
how the method generalises; the submitted models were afterwards retrained on all 75 patients, so
any score of those models on these patients would be in-sample and is deliberately not reported
here. Plan-level local gamma (pymedphys), 10% of prescription cutoff, computed against the
Geant4 ground truth.

## Summary (mean ± SD over the five folds)

| Task | gamma 1%/1mm | gamma 3%/3mm | abdomen 1%/1mm | thorax 1%/1mm |
|---|---|---|---|---|
| Photon-CT | 93.4 ± 1.1 | 99.65 ± 0.15 | 95.2 | 91.8 |
| Photon-MRI | 91.3 ± 1.6 | 99.04 ± 0.40 | 95.4 | 87.5 |
| Proton-CT | 96.9 ± 0.9 | 99.84 ± 0.12 | 98.0 | 95.9 |
| Proton-MRI | 87.4 ± 2.1 | 98.86 ± 0.39 | 91.3 | 83.7 |

## Per fold, gamma 1%/1mm / gamma 3%/3mm

| Task | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| Photon-CT | 94.0 / 99.8 | 92.4 / 99.5 | 94.4 / 99.7 | 94.2 / 99.4 | 92.1 / 99.7 |
| Photon-MRI | 90.7 / 99.1 | 91.2 / 99.0 | 90.4 / 99.1 | 90.2 / 98.5 | 94.1 / 99.6 |
| Proton-CT | 96.7 / 99.8 | 97.5 / 99.9 | 97.4 / 99.9 | 95.5 / 99.6 | 97.4 / 99.9 |
| Proton-MRI | 84.0 / 98.3 | 88.9 / 99.2 | 88.9 / 99.1 | 86.7 / 98.6 | 88.6 / 99.1 |

Per-patient values (all 75 patients x 4 tasks) are in [`cv_results.csv`](cv_results.csv): `task, fold, patient, site, gamma_1pct_1mm, gamma_3pct_3mm`.

Two observations worth keeping in mind when comparing against other work. Thorax is consistently
the harder site, and the gap between the 1%/1mm and 3%/3mm criteria is large: every task is above
98.8% at 3%/3mm, so the errors that fail the strict criterion are small in magnitude and
concentrated at steep dose gradients near lung-tissue interfaces. Hidden-test-set scores of the
submitted models are reported separately in [MODEL_ZOO.md](MODEL_ZOO.md).