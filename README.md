# A Comparative Benchmarking and Hybrid Ensemble Framework for Credit Scoring

A machine-learning credit-scoring research prototype comparing Logistic Regression, XGBoost, and a PyTorch Multi-Layer Perceptron under a common experimental framework.

The project combines the base models using out-of-fold stacking, evaluates probability calibration, translates the final Probability of Default (PD) into scenario-based financial outputs, provides XGBoost SHAP explanations, and deploys the resulting pipeline through Flask, React, and Docker.

## Project Overview

The implemented pipeline is:

```text
Home Credit Data
       │
       ▼
Cleaning & Feature Engineering
       │
       ▼
Model-Specific Preprocessing
       │
       ├── Logistic Regression
       ├── XGBoost
       └── PyTorch MLP
                │
                ▼
       OOF Stacking
                │
                ▼
    Meta Logistic Regression
                │
                ▼
     Probability Calibration
                │
                ▼
            Final PD
          /          \
         ▼            ▼
 Expected Loss    XGBoost SHAP
 Indicative Rate  Explanation
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
- Logistic Regression benchmark
- Weight of Evidence Logistic Regression benchmark
- XGBoost
- PyTorch MLP
- Class weighting and SMOTE experiments
- Out-of-fold stacking
- Logistic Regression meta-learner
- Platt and isotonic probability calibration
- Expected Loss scenario analysis
- Indicative interest-rate prototype
- XGBoost SHAP explanations
- Flask REST API
- React/Vite dashboard
- Docker Compose deployment

## Recorded Results

### Individual Model Validation

| Model | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|
| Logistic Regression | 0.7515 | 0.2342 | 0.0683 |
| XGBoost | **0.7625** | **0.2543** | **0.0675** |
| MLP | 0.7411 | 0.2243 | 0.0692 |

### Final Probability Comparison

| Output | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|
| Raw Stack | 0.7627 | 0.2517 | 0.0682 |
| Platt | 0.7627 | 0.2517 | 0.0683 |
| Isotonic | 0.7618 | 0.2425 | **0.0675** |

Isotonic calibration is used for the deployed final PD because it produced the lowest recorded Brier Score among the compared final probability outputs.

The calibration method was selected after comparison on the final held-out evaluation set; this is documented as a methodological limitation rather than treated as an independent untouched final estimate.

## Documentation

Detailed technical documentation is available in:

- [Methodology](docs/methodology.md) — data preparation, modelling, stacking, calibration, financial framework, and explainability.
- [Experiment Summary](docs/experiments.md) — recorded results, experiments, modelling decisions, and limitations.
- [System Architecture](docs/architecture.md) — saved models, Flask API, React frontend, and Docker deployment.

## Repository Structure

```text
capstone_project_1_fintech/
├── deployment/          # Flask API and model-serving code
├── docs/                # Technical project documentation
├── frontend/            # React/Vite frontend
├── models/              # Saved trained model artifacts
├── notebooks/           # Modelling and analysis notebooks
├── outputs/             # Experiment results and figures
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

## Quick Start with Docker

### Requirements

Install:

- Git
- Docker Desktop

The trained model artifacts are included in the repository. The Home Credit dataset is therefore not required simply to run the completed web application.

### Clone

```bash
git clone https://github.com/dimasputra7/capstone_project_1_fintech.git
cd capstone_project_1_fintech
```

### Start

Make sure Docker Desktop is running, then:

```bash
docker compose up --build
```

Open:

- Frontend: `http://localhost:8080`
- Backend health endpoint: `http://localhost:5000/health`

### Stop

Press `Ctrl + C` in the Docker Compose terminal and run:

```bash
docker compose down
```

After the images have already been built, the application can normally be restarted with:

```bash
docker compose up
```

## Reproducing the Modelling Experiments

The dataset is intentionally excluded from GitHub.

To reproduce the modelling workflow, obtain the Home Credit `application_train.csv` dataset and place it at:

```text
data/application_train.csv
```

The notebooks are organised approximately in workflow order:

```text
notebooks/
├── 01_Data_EDA.ipynb
├── 02_Base_Models.ipynb
├── 03_Stacking_Calibration.ipynb
├── 04_Financial_Analysis.ipynb
├── 05_Explainability.ipynb
└── 06_Deployment.ipynb
```

The notebooks document the experimental workflow, while `models/` contains the serialized artifacts used by the deployed application and `outputs/` contains generated results and figures.

## Financial Framework

Expected Loss is calculated as:

`EL = PD × LGD × EAD`

For this prototype:

- the calibrated ensemble probability is used as PD;
- LGD is a user-controlled scenario assumption;
- `AMT_CREDIT` is used as a proxy for EAD.

The simplified indicative-rate prototype uses:

`Indicative Rate = 0.09 + (PD × LGD)`

These calculations are illustrative research outputs rather than actual lending prices.

## Technology Stack

**Modelling:** Python, pandas, NumPy, scikit-learn, XGBoost, PyTorch

**Explainability:** SHAP

**Backend:** Flask, Flask-CORS

**Frontend:** React, Vite

**Deployment:** Docker, Docker Compose, Nginx

## Limitations

This project is a research prototype. Important limitations include the use of a historical single dataset, class imbalance, simplified LGD/EAD assumptions, simplified indicative pricing, calibration selection using the final evaluation set, XGBoost-specific SHAP explanations, and local rather than public production deployment.

See the [Experiment Summary](docs/experiments.md) for further details.

## Disclaimer

This system is an academic research prototype. Predictions, Expected Loss calculations, SHAP explanations, and indicative interest rates are provided for experimental and educational purposes only.

They should not be interpreted as actual lending decisions, credit offers, financial advice, or production banking policy.