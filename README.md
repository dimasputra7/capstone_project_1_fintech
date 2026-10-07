# A Comparative Benchmarking and Hybrid Ensemble Framework for Credit Scoring

A machine-learning credit-scoring research prototype comparing Logistic Regression, XGBoost, and a PyTorch Multi-Layer Perceptron under a common experimental framework.

The project evaluates multiple modelling approaches, investigates feature-engineering strategies, combines selected models using out-of-fold stacking, applies probability calibration, translates the resulting Probability of Default (PD) into scenario-based financial outputs, provides XGBoost SHAP explanations, and deploys the final pipeline through Flask, React, and Docker.

## Project Overview

The final implemented pipeline is:

```text
Home Credit Data
        │
        ▼
Cleaning & Core Feature Engineering
        │
        ▼
70% Train / 15% Validation / 15% Test
        │
        ▼
Model-Specific Feature Engineering
        │
        ├── LR: Ratios + Log Features
        ├── XGBoost: Ratio Features
        └── MLP: Ratios + Log Features
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
              Final Calibrated PD
                 /          \
                ▼            ▼
        Expected Loss     XGBoost SHAP
        Indicative Rate   Explanation
                 \          /
                  ▼        ▼
                  Flask API
                      │
                      ▼
                React Dashboard
                      │
                      ▼
                    Docker
```

## Key Features

- Home Credit Default Risk dataset
- Logistic Regression statistical benchmark
- Weight of Evidence Logistic Regression reference benchmark
- XGBoost nonlinear benchmark
- PyTorch MLP neural-network benchmark
- Class-weighting and SMOTE experiments
- Employment-normalisation experiment
- Log-transformation experiments
- Financial-ratio feature engineering
- Model-specific feature selection
- Five-fold out-of-fold stacking
- Logistic Regression meta-learner
- Platt and isotonic calibration experiments
- Locked held-out final test evaluation
- Expected Loss scenario analysis
- Indicative interest-rate prototype
- XGBoost SHAP explanations
- Flask REST API
- React/Vite dashboard
- Docker Compose deployment

## Dataset

The project uses the Home Credit Default Risk `application_train.csv` dataset.

- Applications: 307,511
- Original columns: 122
- Default prevalence: approximately 8.07%
- Final modelling predictors: 121
- Numerical predictors: 105
- Categorical predictors: 16

The `DAYS_EMPLOYED = 365243` sentinel value is treated as an anomaly. More interpretable variables including `AGE_YEARS`, `EMPLOYED_YEARS`, and `DAYS_EMPLOYED_ANOM` are created during data preparation.

## Experimental Design

The original dataset was divided using a stratified 70/15/15 split:

| Partition | Observations | Role |
|---|---:|---|
| Training | 215,257 | Initial model development |
| Validation | 46,127 | Model and feature-engineering evaluation |
| Test | 46,127 | Locked final evaluation |

The train and validation partitions were later combined into an approximately 85% development dataset for the final feature-engineered ensemble.

Five-fold out-of-fold predictions were generated within the development data for the three final base models. These probabilities were used as inputs to a Logistic Regression meta-learner.

The held-out 15% test partition was reserved until the final pipeline and calibration strategy had been selected.

## Feature-Engineering Experiments

Feature engineering was evaluated experimentally rather than automatically retained.

### Employment Normalisation

An employment ratio based on employment duration relative to potential working age was evaluated but did not improve the main validation metrics sufficiently and was therefore rejected.

### Log Transformations

Log transformations substantially reduced skewness in several financial variables. Their predictive benefit was modest and model-dependent.

### Financial Ratios

Financial ratios provided the clearest improvement, particularly for XGBoost.

The evaluated ratios included:

- `CREDIT_INCOME_RATIO`
- `ANNUITY_INCOME_RATIO`
- `CREDIT_GOODS_RATIO`
- `ANNUITY_CREDIT_RATIO`

The final model-specific feature configurations were:

| Model | Final Feature Configuration |
|---|---|
| Logistic Regression | Ratios + log-transformed financial features |
| XGBoost | Financial ratios |
| MLP | Ratios + log-transformed financial features |

## Final Locked Test Results

After the feature-engineering, stacking, and calibration decisions were completed using development data, the final pipeline was evaluated on the reserved test partition.

| Model / Output | ROC-AUC | PR-AUC | Brier Score | F1 @ 0.5 | Mean PD |
|---|---:|---:|---:|---:|---:|
| LR — Ratios + Logs | 0.7529 | 0.2280 | 0.06840 | 0.0266 | 0.08079 |
| XGBoost — Ratios | 0.7686 | **0.2564** | **0.06717** | 0.0514 | 0.08033 |
| MLP — Ratios + Logs | 0.7545 | 0.2311 | 0.06827 | 0.0158 | 0.08382 |
| Raw Stack | **0.7688** | 0.2548 | 0.06803 | **0.1181** | 0.08171 |
| Final Isotonic PD | 0.7683 | 0.2452 | 0.06731 | 0.0257 | 0.08094 |

XGBoost was the strongest individual model on the final test set.

The raw stacking ensemble achieved the highest recorded ROC-AUC, while the isotonic output remained the final deployed PD because isotonic calibration had been selected from the development-stage calibration experiment before the locked test set was opened.

The final test results were not subsequently used to retune or replace the selected modelling pipeline.

## Probability Calibration

The project evaluates raw stacking probabilities, Platt scaling, and isotonic regression.

Because the final probability is subsequently used as PD in financial calculations, probability quality was considered alongside discrimination.

Isotonic regression was selected during development based primarily on Brier Score performance and was then frozen as part of the final pipeline before evaluation on the reserved test set.

A methodological limitation remains: the final isotonic mapping was fitted using probabilities from a calibration-stage meta-model and subsequently applied to probabilities produced by the meta-model fitted on the full development data. The two meta-models use the same modelling family and inputs but are not identical fitted models.

A future implementation could use a cleaner cross-fitted or otherwise consistent calibration design.

## Meta-Learner

The final stacking model combines three base-model probabilities:

```text
[LR PD, XGB PD, MLP PD]
            │
            ▼
    Meta Logistic Regression
            │
            ▼
       Raw Ensemble PD
```

The fitted meta-model coefficients were approximately:

| Base Model Probability | Coefficient |
|---|---:|
| Logistic Regression | 0.3622 |
| XGBoost | 5.9710 |
| MLP | 2.4481 |

These coefficients describe how the meta-learner combines the three model probabilities. They should not be interpreted as original-feature importance measures.

## Financial Framework

Expected Loss is calculated as:

`EL = PD × LGD × EAD`

For this prototype:

- PD is the final calibrated ensemble probability;
- LGD is a user-controlled scenario assumption;
- `AMT_CREDIT` is used as a proxy for EAD.

The simplified indicative-rate prototype is:

`Indicative Rate = 0.09 + (PD × LGD)`

The fixed 9% component and LGD scenarios are illustrative research assumptions rather than real lending prices or bank policy.

## Explainability

SHAP is used to provide local explanations for the XGBoost component of the system.

The explanation should be interpreted as:

> Why did the XGBoost component behave this way for this applicant?

It is not an exact explanation of the complete stacked and isotonic-calibrated PD and should not be interpreted causally.

## Application

The completed prototype provides:

- applicant scenario editing;
- baseline-versus-current comparison;
- Logistic Regression probability;
- XGBoost probability;
- MLP probability;
- raw ensemble probability;
- final calibrated PD;
- Expected Loss;
- indicative interest rate;
- XGBoost SHAP drivers;
- LGD scenario analysis.

Applicant characteristics affect model predictions. LGD is handled separately as a financial scenario assumption and therefore changes Expected Loss and the indicative rate without changing PD.

## Repository Structure

```text
capstone_project_1_fintech/
├── deployment/          # Flask API and model-serving code
├── docs/                # Technical project documentation
├── frontend/            # React/Vite frontend
├── models/              # Earlier experimental model artifacts
├── models_final/        # Frozen final feature-engineered artifacts
├── notebooks/           # Modelling and analysis notebooks
├── outputs/             # Experiment results and figures
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

## Documentation

Detailed technical documentation is available in:

- [Methodology](docs/methodology.md) — dataset, preprocessing, feature engineering, modelling, stacking, calibration, financial framework, explainability, and deployment.
- [Experiment Summary](docs/experiments.md) — experiments, results, modelling decisions, and limitations.
- [System Architecture](docs/architecture.md) — final model artifacts, Flask API, React frontend, and Docker architecture.

## Reproducing the Modelling Workflow

The Home Credit dataset is intentionally excluded from GitHub.

To reproduce the modelling experiments, obtain `application_train.csv` and place it at:

```text
data/application_train.csv
```

The notebooks document the development process:

```text
notebooks/
├── 01_Data_EDA.ipynb
├── 02_Base_Models.ipynb
├── 03_Stacking_Calibration.ipynb
├── 04_Financial_Analysis.ipynb
├── 05_Explainability.ipynb
├── 06_Deployment.ipynb
└── 07_Feature_Engineering_Final_Model.ipynb
```

The earlier notebooks preserve the historical experimental workflow. Notebook 07 documents the later feature-engineering experiments and final frozen modelling pipeline.

## Quick Start with Docker

### Requirements

Install:

- Git
- Docker Desktop

The final trained artifacts are included in `models_final/`, so the Home Credit dataset is not required merely to run the completed application.

### Clone

```bash
git clone https://github.com/dimasputra7/capstone_project_1_fintech.git
cd capstone_project_1_fintech
```

### Start

Make sure Docker Desktop is running:

```bash
docker compose up --build
```

Then open:

```text
Frontend:
http://localhost:8080

Backend health:
http://localhost:5000/health
```

### Stop

Press `Ctrl + C`, then run:

```bash
docker compose down
```

After the images have already been built, the application can normally be restarted using:

```bash
docker compose up
```

## Technology Stack

**Data and modelling:** Python, pandas, NumPy, scikit-learn, XGBoost, PyTorch

**Explainability:** SHAP

**Backend:** Flask, Flask-CORS

**Frontend:** React, Vite

**Deployment:** Docker, Docker Compose, Nginx

## Key Limitations

This project is an academic research prototype. Important limitations include:

- evaluation on a single historical public credit dataset;
- substantial class imbalance;
- threshold-dependent F1 behaviour;
- reconstructed fixed MLP architecture for later controlled experiments;
- simplified LGD and EAD assumptions;
- simplified indicative pricing;
- calibration-stage/final-meta-model consistency limitation;
- XGBoost-specific rather than full-ensemble SHAP explanations;
- scenario analysis based on selected editable applicant variables;
- local rather than public production deployment;
- no live bureau or banking integration;
- no production authentication, HTTPS, monitoring, CI/CD, or model-governance framework.

## Research Scope

The project does not propose a new machine-learning algorithm.

Its contribution is the controlled integration and comparison of multiple credit-scoring model families, feature-engineering strategies, out-of-fold stacking, probability calibration, explainability, financial scenario analysis, and containerized deployment under a common experimental framework.

## Disclaimer

This system is an academic research prototype.

Predictions, PD estimates, Expected Loss calculations, SHAP explanations, and indicative interest rates are provided for experimental and educational purposes only.

They should not be interpreted as actual lending decisions, automatic approval/rejection rules, credit offers, financial advice, or production banking policy.