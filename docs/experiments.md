# Experiment Summary

This document records the major experiments, observed results, and modelling decisions made during the project.

## 1. Base Model Benchmark

The main validation comparison was:

| Model | ROC-AUC | PR-AUC | Brier Score | F1 @ 0.5 | Mean PD |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7515 | 0.2342 | 0.0683 | 0.0283 | 0.0809 |
| XGBoost | 0.7625 | 0.2543 | 0.0675 | 0.0384 | 0.0806 |
| MLP | 0.7411 | 0.2243 | 0.0692 | 0.0770 | ~0.0770 |

XGBoost produced the strongest recorded individual validation performance on ROC-AUC, PR-AUC, and Brier Score.

The MLP remained useful as a separate modelling family but did not outperform XGBoost in the recorded validation comparison.

## 2. Weight of Evidence Logistic Regression

A separate WoE Logistic Regression was implemented as a traditional scorecard-style benchmark.

Its role differs from the regular Logistic Regression used by the final ensemble:

- WoE LR: traditional/interpretable benchmark
- Regular LR: probability-producing base model used in stacking

The WoE model was therefore not included as a fourth stacking component.

## 3. Class Imbalance Experiments

### Class Weighting

Class weighting was evaluated to increase the training importance of minority-class defaults.

The experiment altered the probability distribution substantially. Because the project ultimately requires meaningful probability estimates for Expected Loss and indicative-rate calculations, the weighted variants were not retained in the final stack.

### SMOTE

SMOTE was also evaluated.

Synthetic minority observations were restricted to training procedures to prevent leakage into validation or test data.

The observed experiment did not provide sufficient overall improvement to justify retaining SMOTE in the final probability-focused ensemble.

### Decision

The final stacking ensemble therefore uses the untreated versions of:

- Logistic Regression
- XGBoost
- MLP

This was an experimental decision rather than an omission of the proposed imbalance methods.

## 4. Stacking Experiment

Out-of-fold probabilities were generated from Logistic Regression, XGBoost, and MLP.

These three probabilities became the inputs to a Logistic Regression meta-learner:

`[LR PD, XGB PD, MLP PD] → Meta Logistic Regression → Raw Ensemble PD`

OOF prediction was used so that the meta-learner was not trained on base-model predictions produced from observations already seen by those base models.

## 5. Probability Calibration

Three final probability outputs were compared:

| Output | ROC-AUC | PR-AUC | Brier Score | F1 | Mean PD |
|---|---:|---:|---:|---:|---:|
| Raw Stack | 0.7627 | 0.2517 | 0.0682 | 0.1003 | 0.0810 |
| Platt | 0.7627 | 0.2517 | 0.0683 | 0.1100 | 0.0809 |
| Isotonic | 0.7618 | 0.2425 | **0.0675** | 0.0283 | 0.0809 |

Isotonic regression achieved the lowest recorded Brier Score and was therefore selected as the final deployed probability.

This illustrates an important trade-off: the calibration option with the lowest Brier Score did not have the highest discrimination or threshold-dependent F1.

Because PD is subsequently used in financial calculations, probability quality was prioritized for the deployed prototype.

### Calibration Limitation

The calibration methods were compared using the held-out final evaluation set. Therefore, the final test evaluation influenced calibrator selection.

A stronger future design would select calibration using a separate calibration set or nested procedure and reserve another test set for one final unbiased evaluation.

## 6. Financial Scenario Experiments

The final calibrated PD was connected to:

`Expected Loss = PD × LGD × EAD`

with `AMT_CREDIT` used as an EAD proxy.

The prototype also calculates:

`Indicative Rate = 0.09 + (PD × LGD)`

LGD is treated as a scenario assumption rather than a predictive feature.

Therefore:

- changing applicant features may change PD and financial outputs;
- changing only LGD does not change PD;
- changing LGD changes Expected Loss and the indicative rate.

The dashboard supports baseline-versus-scenario comparison to demonstrate these effects.

## 7. SHAP Explainability

SHAP analysis was implemented for the XGBoost component.

The project contains both global and local SHAP outputs, including feature-importance and applicant-level explanation figures.

SHAP results are interpreted as explanations of XGBoost behaviour, not causal effects and not exact explanations of the complete calibrated stacking ensemble.

## 8. Final Experimental Decisions

| Decision | Rationale |
|---|---|
| Split data before learned preprocessing/SMOTE | Reduce leakage and preserve realistic validation/test distributions |
| Keep untreated base models in final stack | Imbalance interventions did not provide sufficient overall benefit for the probability-focused system |
| Use OOF predictions for stacking | Prevent meta-learner training on in-sample base predictions |
| Use Logistic Regression as meta-learner | Simple second-level combination of base probabilities |
| Use isotonic-calibrated final PD | Lowest recorded Brier Score among final calibration comparisons |
| Use `AMT_CREDIT` as EAD proxy | Separate EAD model unavailable in the prototype |
| Treat LGD as scenario assumption | Enables sensitivity analysis without changing predicted PD |
| Use SHAP for XGBoost | Efficient tree-based local explanation |
| Deploy using Flask + React + Docker | Provides an integrated reproducible application prototype |

## 9. Key Limitations

- Results are based on the historical Home Credit dataset and should not automatically be generalized to other populations or lenders.
- The dataset is highly imbalanced.
- Threshold-based metrics depend on the selected classification threshold.
- Calibration selection used the final evaluation set.
- `AMT_CREDIT` is an EAD proxy rather than an estimated EAD.
- LGD is a scenario assumption rather than an estimated recovery model.
- The indicative-rate equation is deliberately simplified.
- SHAP explains XGBoost rather than the complete final ensemble.
- The UI exposes selected scenario variables while remaining model features retain baseline values.
- Public cloud deployment, production authentication, HTTPS, monitoring, model governance, and CI/CD are outside the implemented scope.