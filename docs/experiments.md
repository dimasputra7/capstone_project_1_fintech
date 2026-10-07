# Experiment Summary

This document records the major experiments, observed results, modelling decisions, and final evaluation of the credit-scoring project.

The project followed an iterative experimental process. Earlier benchmark experiments were retained as development evidence, while later feature-engineering experiments were conducted in response to supervisor feedback and used to construct the final frozen modelling pipeline.

## 1. Original Base Model Benchmark

Three main modelling families were initially evaluated under a common experimental framework:

| Model | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|
| Logistic Regression | 0.7515 | 0.2342 | 0.0683 |
| XGBoost | **0.7625** | **0.2543** | **0.0675** |
| MLP | 0.7411 | 0.2243 | 0.0692 |

XGBoost produced the strongest original individual benchmark performance.

The MLP did not outperform XGBoost, but it was retained because it represents a distinct neural-network modelling family and provides additional model diversity for the ensemble.

## 2. Weight of Evidence Logistic Regression

A separate Weight of Evidence (WoE) Logistic Regression model was implemented as a traditional scorecard-style benchmark.

Its role differs from the regular Logistic Regression used by the ensemble:

- WoE LR: traditional scorecard-style reference model.
- Regular LR: probability-producing statistical base model used in stacking.

The WoE model was therefore not included as a fourth stacking component.

## 3. Class-Imbalance Experiments

### 3.1 Class Weighting

Class weighting was evaluated to increase the training importance of minority-class defaults.

Although weighting can improve sensitivity to the minority class, the experiment substantially altered probability behaviour.

Because the project ultimately uses estimated Probability of Default in financial calculations, probability quality was important in addition to classification performance.

The weighted variants were therefore not retained in the final probability-focused pipeline.

### 3.2 SMOTE

SMOTE was also evaluated.

Synthetic minority observations were restricted to training procedures so that synthetic data did not leak into validation or test partitions.

The experiment did not provide sufficient overall benefit to justify retaining SMOTE in the final pipeline.

### 3.3 Decision

Class weighting and SMOTE were therefore treated as evaluated experimental alternatives rather than mandatory components of the final model.

## 4. Supervisor-Feedback Feature-Engineering Stage

Following feedback on the initial modelling framework, additional feature-engineering experiments were performed.

The purpose was not simply to create more variables, but to determine experimentally whether financially meaningful transformations improved model performance.

Three main groups were investigated:

1. employment normalisation;
2. log transformations of skewed financial variables;
3. financial ratio features.

Feature-engineering decisions were based on comparative validation results rather than assumptions that additional engineered variables would automatically improve the models.

## 5. Employment-Ratio Experiment

An employment-normalisation feature was constructed using employment duration relative to potential working age.

Conceptually:

`EMPLOYMENT_RATIO = EMPLOYED_YEARS / (AGE_YEARS - 18)`

The ratio was calculated only where the underlying employment information was considered valid. The original `DAYS_EMPLOYED = 365243` anomaly remained handled separately.

### Results

| Configuration | ROC-AUC | PR-AUC | Brier Score | F1 |
|---|---:|---:|---:|---:|
| Baseline | 0.75736 | 0.24901 | 0.067659 | 0.04099 |
| Employment Ratio | 0.75709 | 0.24760 | 0.067706 | 0.03839 |

The engineered employment ratio did not improve the evaluated metrics.

### Decision

**Rejected.**

The experiment demonstrates that a feature can be conceptually meaningful without providing additional predictive value.

## 6. Log-Transformation Experiments

Several financial variables were strongly right-skewed.

Log transformations were evaluated to reduce skewness and potentially provide more suitable representations, particularly for models sensitive to feature scale or linear relationships.

Approximate skewness changes included:

| Variable | Original Skewness | Log-Transformed Skewness |
|---|---:|---:|
| Income | 391.56 | 0.17 |
| Credit | 1.23 | -0.34 |
| Annuity | 1.58 | -0.35 |
| Goods Price | 1.35 | -0.29 |

The transformations substantially changed the statistical distributions, but predictive improvements were model-dependent.

### Logistic Regression

| Configuration | ROC-AUC | PR-AUC | Brier Score | F1 | Mean PD |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.748083 | 0.230662 | 0.068462 | 0.027282 | 0.080843 |
| Log Income | 0.747991 | 0.230622 | 0.068464 | 0.026247 | 0.080834 |
| All Logs | **0.749224** | **0.232986** | **0.068357** | **0.030374** | 0.080810 |

### XGBoost

| Configuration | ROC-AUC | PR-AUC | Brier Score | F1 | Mean PD |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.757062 | 0.247693 | 0.067716 | 0.037843 | 0.080319 |
| Log Income | **0.757292** | **0.247979** | **0.067694** | **0.040890** | 0.080485 |
| All Logs | 0.756453 | 0.247515 | 0.067729 | 0.040467 | 0.080409 |

### Decision

Log transformations were **not universally beneficial**.

They were useful as part of the final Logistic Regression and MLP feature configurations, but the full set of log transformations was not retained for XGBoost.

## 7. Financial-Ratio Experiments

Four financially meaningful ratios were evaluated:

- `CREDIT_INCOME_RATIO`
- `ANNUITY_INCOME_RATIO`
- `CREDIT_GOODS_RATIO`
- `ANNUITY_CREDIT_RATIO`

These variables express relationships between financial quantities rather than considering each amount independently.

### Logistic Regression

| Configuration | ROC-AUC | PR-AUC | Brier Score | F1 | Mean PD |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.748083 | 0.230662 | 0.068462 | 0.027282 | 0.080843 |
| All Ratios | 0.749519 | 0.232827 | 0.068357 | 0.029820 | 0.080845 |
| Ratios + All Logs | **0.750621** | **0.235055** | **0.068263** | **0.031307** | 0.080805 |

The combined ratios-and-logs configuration was retained for Logistic Regression.

### XGBoost

| Configuration | ROC-AUC | PR-AUC | Brier Score | F1 | Mean PD |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.757062 | 0.247693 | 0.067716 | 0.037843 | 0.080319 |
| All Ratios | **0.765558** | 0.255163 | 0.067298 | 0.048831 | 0.080204 |
| Ratios + Log Income | 0.765521 | **0.255914** | **0.067273** | **0.051755** | 0.080260 |

Financial ratios produced a clear improvement for XGBoost.

`ANNUITY_CREDIT_RATIO` was the strongest individual ratio observed in the XGBoost ratio experiments.

For the final controlled pipeline, the ratio-based XGBoost configuration was retained.

## 8. MLP Feature-Engineering Experiment

The later controlled MLP experiments used a fixed reconstructed architecture:

```text
Input
  ↓
128 units
  ↓
ReLU
  ↓
Dropout (0.20)
  ↓
64 units
  ↓
ReLU
  ↓
Dropout (0.20)
  ↓
Output
```

Training used `BCEWithLogitsLoss` and the Adam optimizer.

This architecture was held fixed so that the later comparison focused on feature representation rather than simultaneous architecture tuning.

### Results

| Configuration | ROC-AUC | PR-AUC | Brier Score | F1 | Mean PD |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.748796 | 0.230121 | 0.068447 | 0.001609 | 0.081875 |
| Ratios | 0.751293 | 0.237145 | 0.068231 | **0.010125** | 0.081184 |
| Ratios + Logs | **0.752253** | **0.240995** | **0.068155** | 0.001610 | 0.077509 |

### Decision

**Ratios + logs retained.**

The exact earliest historical MLP configuration was not available for complete reconstruction. Therefore, these later MLP experiments should be interpreted as controlled comparisons using a fixed reconstructed architecture rather than exact reproductions of every earlier MLP experiment.

## 9. Final Model-Specific Feature Configurations

The feature-engineering experiments resulted in different configurations for the three modelling families.

| Model | Retained Configuration |
|---|---|
| Logistic Regression | Financial ratios + log transformations |
| XGBoost | Financial ratios |
| MLP | Financial ratios + log transformations |

This model-specific approach was preferred over forcing the same engineered representation onto all three models.

## 10. Feature-Engineered Stacking Experiment

The three feature-engineered models were then combined using out-of-fold stacking.

The general structure was:

```text
LR (Ratios + Logs)
        │
XGB (Ratios)
        │
MLP (Ratios + Logs)
        │
        ▼
5-Fold OOF Probabilities
        │
        ▼
Meta Logistic Regression
        │
        ▼
Raw Ensemble PD
```

### Development Comparison

| Stack | ROC-AUC | PR-AUC | Brier Score | F1 | Mean PD |
|---|---:|---:|---:|---:|---:|
| Baseline Stack | 0.754343 | 0.241684 | 0.068609 | 0.100928 | 0.081318 |
| Feature-Engineered Stack | **0.762650** | **0.248052** | **0.068323** | **0.106317** | 0.079581 |

Approximate changes were:

- ROC-AUC: +0.00831
- PR-AUC: +0.00637
- Brier Score: -0.000286
- F1: +0.00539

### Decision

The feature-engineered stack was retained for the final modelling pipeline.

## 11. Development-Stage Calibration Experiment

Calibration alternatives were compared during development before opening the final held-out test partition.

### Baseline Stack

| Output | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|
| Raw | 0.761572 | **0.249195** | 0.068227 |
| Platt | 0.761572 | **0.249195** | 0.069552 |
| Isotonic | 0.760955 | 0.239098 | **0.067603** |

### Feature-Engineered Stack

| Output | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|
| Raw | **0.768947** | **0.255870** | 0.067943 |
| Platt | **0.768947** | **0.255870** | 0.069377 |
| Isotonic | 0.768072 | 0.246190 | **0.067118** |

Isotonic regression achieved the lowest development-stage Brier Score for the feature-engineered stack.

Because the final probability is intended to represent PD and is subsequently used in financial calculations, probability quality was prioritized.

### Decision

**Isotonic regression was selected before the final test set was opened.**

The selected calibration approach was then frozen as part of the final pipeline.

## 12. Final Frozen Pipeline

The final modelling configuration was:

```text
LR — Ratios + Logs
        │
XGBoost — Ratios
        │
MLP — Ratios + Logs
        │
        ▼
5-Fold OOF Stacking
        │
        ▼
Meta Logistic Regression
        │
        ▼
Isotonic Calibration
        │
        ▼
Final PD
```

The original training and validation partitions were combined into an approximately 85% development dataset.

Five-fold OOF predictions were generated within this development set.

The original 15% test partition remained reserved until the modelling and calibration decisions had been made.

## 13. Final Locked Test Evaluation

The final frozen pipeline was evaluated once on the reserved test partition containing 46,127 observations.

| Model / Output | ROC-AUC | PR-AUC | Brier Score | F1 @ 0.5 | Mean PD |
|---|---:|---:|---:|---:|---:|
| LR — Ratios + Logs | 0.752869 | 0.228018 | 0.068404 | 0.026646 | 0.080790 |
| XGBoost — Ratios | 0.768588 | **0.256410** | **0.067168** | 0.051374 | 0.080327 |
| MLP — Ratios + Logs | 0.754493 | 0.231109 | 0.068268 | 0.015835 | 0.083818 |
| Raw Stack | **0.768823** | 0.254761 | 0.068026 | **0.118064** | 0.081715 |
| Final Isotonic | 0.768251 | 0.245151 | 0.067306 | 0.025729 | 0.080942 |

The test default prevalence was approximately 8.07%.

### Interpretation

XGBoost was the strongest individual model on the final held-out test data.

The raw stack achieved the highest ROC-AUC, while XGBoost achieved the strongest PR-AUC and lowest Brier Score among the final test outputs.

The isotonic output remained the deployed final PD because the calibration method had already been selected during development. The final test results were not used to replace or retune the frozen pipeline.

This distinction is important: the purpose of the locked test set was evaluation rather than another round of model selection.

## 14. F1 at the 0.5 Threshold

The final isotonic output has a lower F1 score at the default 0.5 classification threshold.

This does not automatically imply poor probability estimation.

F1 is threshold-dependent, whereas the project primarily uses the model output as a continuous Probability of Default.

Calibration can remap probabilities without materially changing the underlying ranking of applicants, causing the number of observations exceeding a fixed threshold such as 0.5 to change substantially.

For this reason, ROC-AUC, PR-AUC, Brier Score, probability behaviour, and the intended PD use case are considered alongside F1.

No new threshold was selected using the final test set.

## 15. Meta-Learner

The fitted final meta-logistic coefficients were approximately:

| Input Probability | Coefficient |
|---|---:|
| Logistic Regression | 0.3622 |
| XGBoost | 5.9710 |
| MLP | 2.4481 |

The relatively large XGBoost coefficient is consistent with the strong contribution of the XGBoost probability to the ensemble.

However, these coefficients are **meta-model combination weights**, not feature importance measures for the original applicant variables.

## 16. Calibration Implementation Limitation

The final calibration procedure contains an implementation limitation that should be stated explicitly.

During the development procedure, a meta-model was fitted for the calibration-stage probability generation and isotonic regression was fitted to those probabilities.

The final raw test probabilities were subsequently produced using a meta-logistic model fitted on the full development data.

Therefore, the isotonic mapping is applied to probabilities from a meta-model that uses the same model family and meta-features but is not the identical fitted meta-model used to generate the calibration probabilities.

This does not justify changing the model after the locked test evaluation.

A cleaner future design could use:

- a consistent meta-model/calibrator mapping;
- cross-fitted calibration;
- nested validation; or
- a dedicated calibration partition followed by a completely untouched test set.

The limitation is documented rather than hidden.

## 17. Financial Scenario Framework

The final calibrated PD is connected to:

`Expected Loss = PD × LGD × EAD`

where `AMT_CREDIT` is used as an EAD proxy.

The prototype also calculates:

`Indicative Rate = 0.09 + (PD × LGD)`

LGD is treated as a scenario assumption rather than a predictive model input.

Therefore:

- changing applicant characteristics may change PD;
- changing PD changes Expected Loss and the indicative rate;
- changing only LGD does not change PD;
- changing LGD changes Expected Loss and the indicative rate.

These outputs are illustrative research calculations rather than real lending prices or policies.

## 18. SHAP Explainability

SHAP is implemented for the XGBoost component.

The deployed application provides local XGBoost feature contributions for applicant-level scenario analysis.

These values explain the behaviour of the XGBoost component.

They should **not** be interpreted as:

- causal effects;
- explanations of the Logistic Regression component;
- explanations of the MLP component; or
- an exact decomposition of the final stacked and calibrated PD.

## 19. Final Experimental Decisions

| Decision | Rationale |
|---|---|
| Split before learned preprocessing and SMOTE | Reduce leakage |
| Evaluate weighting and SMOTE but exclude them from final pipeline | Insufficient benefit for probability-focused objective |
| Reject employment ratio | Did not improve validation performance |
| Retain ratios + logs for LR | Best controlled LR feature configuration |
| Retain ratios for XGBoost | Clear improvement from financial ratios |
| Retain ratios + logs for MLP | Strongest controlled MLP configuration |
| Use model-specific feature engineering | Different model families responded differently to transformations |
| Use 5-fold OOF stacking | Reduce meta-learning leakage |
| Use Logistic Regression meta-learner | Simple probability-level second-stage model |
| Select isotonic during development | Lowest development Brier Score among calibration alternatives |
| Freeze final test after evaluation | Avoid test-driven retuning |
| Use `AMT_CREDIT` as EAD proxy | Separate EAD model unavailable |
| Treat LGD as scenario assumption | Enables sensitivity analysis without changing PD |
| Use SHAP for XGBoost | Provides efficient tree-based local explanation |
| Deploy with Flask + React + Docker | Integrated reproducible prototype |

## 20. Final Model Artifacts

The frozen deployment artifacts are stored in:

```text
models_final/
├── feature_config.json
├── isotonic_calibrator.joblib
├── lr_model.joblib
├── lr_preprocessor.joblib
├── meta_logistic.joblib
├── mlp_metadata.json
├── mlp_preprocessor.joblib
├── mlp_state.pt
├── xgb_model.joblib
└── xgb_preprocessor.joblib
```

These artifacts correspond to the final feature-engineered deployment pipeline.

## 21. Key Limitations

The main experimental limitations are:

- results are based on one historical public credit dataset;
- the target remains highly imbalanced;
- the later MLP architecture is a fixed reconstructed architecture rather than an exact reconstruction of every earlier MLP experiment;
- threshold-based metrics such as F1 depend on the selected classification threshold;
- the calibration-stage and final full-development meta-models are not identical fitted models;
- `AMT_CREDIT` is a proxy rather than a separately estimated EAD;
- LGD is a scenario assumption rather than an estimated recovery model;
- the indicative-rate equation is deliberately simplified;
- SHAP explains the XGBoost component rather than the complete ensemble;
- no statistical-significance claim is made for small metric differences;
- the UI exposes selected applicant variables while other model inputs retain baseline values;
- public production infrastructure, authentication, monitoring, CI/CD, fairness governance, and live banking integration are outside the implemented scope.

## 22. Model Freeze

Following the final held-out evaluation, the modelling pipeline was frozen.

No additional feature selection, hyperparameter tuning, calibration selection, or threshold optimisation should be performed using the final test results.

Further modelling changes should only be considered if:

1. a genuine implementation error is discovered that invalidates the experiment; or
2. new supervisor feedback explicitly requires an additional experiment.

Any such future experiment should be clearly documented as occurring after the current frozen final evaluation.