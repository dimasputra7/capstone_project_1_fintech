# Methodology

## 1. Overview

This project implements a comparative credit-scoring framework using the Home Credit Default Risk dataset.

The objective is to estimate Probability of Default (PD) under a common experimental environment, compare multiple modelling approaches, evaluate feature-engineering strategies, combine selected models through out-of-fold stacking, calibrate the resulting probability, and connect the final PD to an illustrative financial risk framework.

The final workflow is:

```text
Home Credit Data
        ↓
Cleaning and Core Feature Engineering
        ↓
Stratified Train / Validation / Test Split
        ↓
Model-Specific Preprocessing
        ↓
Feature-Engineering Experiments
        ↓
LR + XGBoost + MLP
        ↓
5-Fold OOF Stacking
        ↓
Meta Logistic Regression
        ↓
Probability Calibration
        ↓
Final PD
        ↓
Financial Analysis + Explainability
        ↓
Flask API
        ↓
React Dashboard
        ↓
Docker
```

## 2. Dataset

The implementation uses `application_train.csv` from the Home Credit Default Risk dataset.

Dataset characteristics:

- Observations: 307,511
- Original columns: 122
- Target: `TARGET`
- Default prevalence: approximately 8.07%
- Non-default prevalence: approximately 91.93%

The target is substantially imbalanced.

For this reason, accuracy is not treated as the primary evaluation metric. Model evaluation considers:

- ROC-AUC
- PR-AUC
- Brier Score
- F1 Score
- Mean predicted PD

ROC-AUC and PR-AUC evaluate discrimination, while Brier Score is particularly relevant because the final model output is intended to be interpreted as a probability.

## 3. Initial Data Cleaning

The Home Credit variable `DAYS_EMPLOYED` contains the sentinel value:

`365243`

This value was treated as an anomalous code rather than a genuine employment duration.

The implementation creates more interpretable variables including:

- `AGE_YEARS`
- `EMPLOYED_YEARS`
- `DAYS_EMPLOYED_ANOM`

The original identifiers and replaced duration variables are excluded from the final predictor set where appropriate.

The final modelling dataset contains:

- 121 predictors
- 105 numerical predictors
- 16 categorical predictors

## 4. Data Partitioning and Leakage Control

The data was divided using a stratified 70/15/15 split.

| Partition | Observations | Approximate Share |
|---|---:|---:|
| Training | 215,257 | 70% |
| Validation | 46,127 | 15% |
| Test | 46,127 | 15% |

The split was performed before learned preprocessing or synthetic sampling procedures.

This was intended to reduce information leakage between development and evaluation data.

During the initial experimental stage:

- training data was used for model fitting;
- validation data was used for model and feature-engineering comparisons;
- test data remained reserved.

For the final modelling stage, the original training and validation partitions were combined into an approximately 85% development dataset.

The original 15% test partition remained reserved for the final locked evaluation.

## 5. Model-Specific Preprocessing

Separate preprocessing pipelines are maintained for the three modelling families.

This allows each model to use a consistent transformation process during both training and deployment.

The final serialized deployment therefore contains separate preprocessors for:

- Logistic Regression
- XGBoost
- MLP

Preprocessing objects are fitted using development data and saved with the corresponding trained models.

## 6. Base Modelling Families

Three primary modelling families are evaluated.

### 6.1 Logistic Regression

Logistic Regression provides the statistical and linear benchmark.

It produces probability estimates directly and provides a useful contrast with more flexible nonlinear approaches.

A separate Weight of Evidence Logistic Regression implementation was also evaluated as a traditional scorecard-style reference.

The WoE model is not a component of the final stack.

### 6.2 XGBoost

XGBoost provides the nonlinear boosted-tree modelling family.

It can capture nonlinear relationships and feature interactions that are not directly represented by a linear Logistic Regression model.

XGBoost is also used as the explainable component of the deployed system through tree-based SHAP analysis.

### 6.3 Multi-Layer Perceptron

A PyTorch Multi-Layer Perceptron provides the neural-network modelling family.

The later controlled feature-engineering experiments use the fixed architecture:

```text
Input
  ↓
128
  ↓
ReLU
  ↓
Dropout (0.20)
  ↓
64
  ↓
ReLU
  ↓
Dropout (0.20)
  ↓
Output
```

Training uses:

- `BCEWithLogitsLoss`
- Adam optimizer
- learning rate = 0.001
- batch size = 2048
- 12 epochs

The architecture was held fixed during the later feature-engineering comparison so that differences could be attributed primarily to feature representation rather than simultaneous neural-network tuning.

The exact earliest historical MLP configuration was not available for complete reconstruction. Therefore, later MLP experiments are treated as controlled experiments using a fixed reconstructed architecture.

## 7. Class-Imbalance Experiments

The project evaluates both class weighting and SMOTE.

These techniques are treated as experimental alternatives rather than automatically assumed to improve the final model.

### Class Weighting

Class weighting increases the training importance of minority-class default observations.

The experiments showed that weighting could substantially alter probability behaviour.

Because the final system uses predicted probabilities as PD, probability quality was important in addition to classification performance.

### SMOTE

SMOTE creates synthetic minority-class training observations.

Where SMOTE was evaluated, synthetic observations were restricted to training procedures and were not introduced into validation or test partitions.

### Final Decision

Neither class weighting nor SMOTE provided sufficient overall benefit for the probability-focused final system.

They were therefore evaluated but not retained in the final feature-engineered pipeline.

## 8. Feature-Engineering Methodology

Additional feature-engineering experiments were conducted after the initial model benchmarking stage.

Three main approaches were evaluated:

1. employment normalisation;
2. log transformations;
3. financial ratios.

Features were retained only when the controlled comparisons supported their use.

### 8.1 Employment Normalisation

An employment ratio was evaluated using:

`EMPLOYMENT_RATIO = EMPLOYED_YEARS / (AGE_YEARS - 18)`

The purpose was to represent employment duration relative to an applicant's approximate potential working life.

The feature did not improve the principal validation metrics and was therefore rejected.

### 8.2 Log Transformations

Several monetary variables showed substantial positive skew.

Log transformations were evaluated for financial variables including:

- income;
- credit amount;
- annuity;
- goods price.

The transformations substantially reduced skewness, but their predictive value differed across model families.

They were therefore retained selectively rather than applied universally.

### 8.3 Financial Ratios

Financial ratios were engineered to represent relationships between applicant financial quantities.

The main ratios were:

- `CREDIT_INCOME_RATIO`
- `ANNUITY_INCOME_RATIO`
- `CREDIT_GOODS_RATIO`
- `ANNUITY_CREDIT_RATIO`

These ratios were evaluated separately and in combination with log transformations.

The ratio experiments produced particularly clear improvements for XGBoost.

## 9. Final Model-Specific Feature Configurations

The feature-engineering experiments resulted in different retained representations for the three modelling families.

| Model | Final Configuration |
|---|---|
| Logistic Regression | Ratios + log transformations |
| XGBoost | Financial ratios |
| MLP | Ratios + log transformations |

The final methodology therefore uses **model-specific feature engineering** rather than requiring all base models to use identical engineered feature sets.

The original underlying applicant data remains common across the modelling framework.

## 10. Out-of-Fold Stacking

The final ensemble combines the three feature-engineered base models through five-fold out-of-fold prediction.

The final base models are:

```text
Logistic Regression
(Ratios + Logs)

XGBoost
(Ratios)

MLP
(Ratios + Logs)
```

For each fold, a base model is trained on the remaining development folds and predicts probabilities for observations in the held-out fold.

As a result, each development observation receives a probability from a model that was not trained on that observation.

The three OOF probabilities become the meta-features:

```text
[LR PD, XGB PD, MLP PD]
```

These are supplied to a Logistic Regression meta-learner:

```text
LR PD ─────┐
XGB PD ────┼──→ Meta Logistic Regression ──→ Raw Ensemble PD
MLP PD ────┘
```

OOF prediction reduces the risk of training the meta-learner on overly optimistic in-sample base-model predictions.

## 11. Meta-Learner

Logistic Regression is used as the second-level meta-model.

The final fitted coefficients are approximately:

| Meta-Feature | Coefficient |
|---|---:|
| LR probability | 0.3622 |
| XGBoost probability | 5.9710 |
| MLP probability | 2.4481 |

These coefficients describe the combination of base-model probabilities.

They are not feature-importance measures for the original applicant variables.

## 12. Probability Calibration

Because the final model output is subsequently interpreted as PD and used in financial calculations, probability quality is considered explicitly.

Three probability outputs were evaluated during development:

- raw stacking probability;
- Platt scaling;
- isotonic regression.

For the feature-engineered stack, the development-stage comparison was:

| Output | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|
| Raw | 0.768947 | 0.255870 | 0.067943 |
| Platt | 0.768947 | 0.255870 | 0.069377 |
| Isotonic | 0.768072 | 0.246190 | **0.067118** |

Isotonic regression produced the lowest development-stage Brier Score.

It was therefore selected as the final calibration method **before the reserved test partition was opened**.

The selected calibration method was then frozen as part of the final pipeline.

## 13. Final Development Procedure

For the final pipeline, the original training and validation partitions were combined into the approximately 85% development dataset.

Five-fold OOF predictions were generated across this development data.

The OOF meta-data was subsequently divided for the meta-learning/calibration procedure using `random_state=52`.

A calibration-stage meta-logistic model generated the probabilities used to fit the isotonic mapping.

The final base models were then trained using the full development dataset.

A final meta-logistic model was also fitted using the complete development OOF meta-data.

The resulting deployment path is:

```text
Applicant
   ↓
Model-Specific Feature Engineering
   ↓
LR + XGBoost + MLP
   ↓
Three Base Probabilities
   ↓
Final Meta Logistic Regression
   ↓
Raw Ensemble PD
   ↓
Isotonic Mapping
   ↓
Final Calibrated PD
```

## 14. Calibration Implementation Limitation

The final calibration implementation has an important methodological limitation.

The isotonic calibrator was fitted to probabilities produced by the calibration-stage meta-logistic model.

During final inference, however, the raw probabilities are produced by the meta-logistic model fitted on the full development OOF data.

The two meta-models:

- use the same modelling family;
- use the same three probability inputs;
- are trained within the same overall development framework;

but they are not the identical fitted model.

Therefore, the isotonic mapping is being applied to probabilities from a closely related but separately fitted meta-model.

This limitation is documented rather than corrected after inspecting the final test results.

A future implementation could improve this procedure using a consistent calibration mapping, cross-fitted calibration, nested validation, or a dedicated calibration design.

## 15. Locked Final Evaluation

After the model-specific feature configurations, stacking procedure, and isotonic calibration strategy had been selected, the reserved 15% test partition was opened for final evaluation.

The test set contains:

`46,127 observations`

The final test results were:

| Model / Output | ROC-AUC | PR-AUC | Brier Score | F1 @ 0.5 |
|---|---:|---:|---:|---:|
| LR — Ratios + Logs | 0.752869 | 0.228018 | 0.068404 | 0.026646 |
| XGBoost — Ratios | 0.768588 | **0.256410** | **0.067168** | 0.051374 |
| MLP — Ratios + Logs | 0.754493 | 0.231109 | 0.068268 | 0.015835 |
| Raw Stack | **0.768823** | 0.254761 | 0.068026 | **0.118064** |
| Final Isotonic | 0.768251 | 0.245151 | 0.067306 | 0.025729 |

These results are treated as final evaluation evidence rather than another model-selection stage.

No further tuning was performed based on these test results.

## 16. Interpretation of Threshold-Dependent F1

F1 is reported at the conventional threshold of 0.5.

The project does not optimize a new classification threshold using the final test set.

This is important because the primary system output is a continuous PD rather than simply a binary classification.

Calibration can change the numerical probability scale and therefore alter the number of applicants exceeding 0.5 without necessarily producing a comparable change in ranking performance.

F1 is therefore interpreted alongside ROC-AUC, PR-AUC, Brier Score, and the intended PD use case.

## 17. Financial Framework

The final calibrated PD is translated into an illustrative Expected Loss calculation:

`Expected Loss = PD × LGD × EAD`

where:

- PD = final calibrated Probability of Default;
- LGD = Loss Given Default scenario assumption;
- EAD = Exposure at Default.

The prototype uses `AMT_CREDIT` as a proxy for EAD rather than estimating a separate EAD model.

LGD is controlled independently as a scenario assumption.

Changing LGD therefore changes the financial outputs but does not change the model-generated PD.

## 18. Indicative Rate

The prototype calculates:

`Indicative Rate = 0.09 + (PD × LGD)`

The fixed 9% component is an illustrative research assumption representing simplified non-PD pricing components.

The result is **not** intended to represent an actual bank interest-rate policy, credit offer, or production pricing model.

## 19. Explainability

SHAP is used to provide local feature-level explanations for the XGBoost component.

The applicant is transformed using the XGBoost preprocessing pipeline and evaluated using a tree-based SHAP explainer.

The explanation therefore describes the XGBoost component only.

It is not:

- an exact decomposition of the final stacked probability;
- an exact explanation of the isotonic calibration;
- a causal explanation of default risk.

The SHAP output is used as decision-support information for understanding influential model signals.

## 20. Deployment Methodology

The final pipeline is deployed through:

- Flask REST API;
- React/Vite frontend;
- Docker;
- Docker Compose;
- Nginx for frontend serving.

The final trained artifacts are stored under:

```text
models_final/
```

Separate serialized preprocessors and models are loaded by the backend so that new applicant records are transformed consistently with the final modelling workflow.

The application exposes:

```text
GET /health
POST /predict
```

The prediction service returns:

- LR probability;
- XGBoost probability;
- MLP probability;
- raw ensemble probability;
- final calibrated PD;
- Expected Loss;
- indicative rate;
- XGBoost SHAP information.

## 21. Scenario Analysis

The React dashboard starts from a complete baseline applicant record.

A selected subset of applicant variables can be modified interactively.

Variables not exposed in the interface retain their baseline values.

The dashboard compares:

```text
Baseline Applicant
        vs
Current Scenario
```

This allows the prototype to demonstrate how changes in applicant characteristics affect predicted PD and downstream financial outputs.

LGD is controlled separately because it is a financial scenario assumption rather than a predictive applicant feature.

## 22. Model Freeze

Following the locked test evaluation, the current modelling pipeline is treated as frozen.

No additional:

- feature selection;
- hyperparameter tuning;
- calibration selection;
- model replacement; or
- threshold optimisation

should be performed using the final test results.

Additional modelling work should only occur if a genuine implementation problem is discovered or new project feedback explicitly requires another experiment.

## 23. Methodological Scope

The project does not claim to introduce a new machine-learning algorithm.

Its methodological contribution is the controlled integration and evaluation of:

```text
Multiple Model Families
        +
Feature Engineering
        +
OOF Stacking
        +
Probability Calibration
        +
Financial Interpretation
        +
Explainability
        +
Application Deployment
```

within a common credit-scoring experimental framework.