# Credit Scoring Capstone Project

## A Comparative Benchmarking and Hybrid Ensemble Framework for Credit Scoring

This project develops and evaluates a machine-learning credit scoring framework using the Home Credit Default Risk dataset.

The project compares multiple credit-risk modelling approaches and combines selected models through a stacking ensemble. The final probability of default (PD) is calibrated and used in an illustrative financial risk framework for expected loss and indicative interest-rate estimation.

## Project Workflow

1. Data cleaning and exploratory data analysis
2. Feature engineering and preprocessing
3. Train/validation/test preparation
4. Logistic Regression benchmarking
5. Weight of Evidence (WoE) Logistic Regression benchmarking
6. Class imbalance experiments
7. XGBoost modelling
8. PyTorch Multi-Layer Perceptron (MLP)
9. Out-of-fold prediction generation
10. Logistic Regression stacking ensemble
11. Probability calibration
12. Final model evaluation
13. Expected loss analysis
14. Indicative interest-rate calculation
15. SHAP explainability
16. Flask REST API
17. React user interface
18. Docker deployment

## Models

The main predictive models investigated are:

- Logistic Regression
- WoE Logistic Regression
- XGBoost
- Multi-Layer Perceptron (PyTorch)

The final stacking framework combines predictions from:

- Logistic Regression
- XGBoost
- MLP

A Logistic Regression meta-learner combines the base-model probabilities.

## Probability Calibration

Raw ensemble probabilities are evaluated alongside:

- Platt scaling
- Isotonic regression

The deployed framework uses the selected calibrated probability as the final probability of default (PD).

## Financial Risk Framework

The calibrated PD is used to calculate illustrative financial risk measures.

Expected Loss:

EL = PD × LGD × EAD

where:

- PD = Probability of Default
- LGD = Loss Given Default
- EAD = Exposure at Default

For this prototype, the loan credit amount (`AMT_CREDIT`) is used as a proxy for EAD.

The system also produces an illustrative indicative interest rate based on the estimated credit risk.

These outputs are intended for research and demonstration purposes only and do not represent actual lending decisions or loan offers.

## Explainability

SHAP is used to explain the XGBoost component of the framework and identify features contributing to individual risk predictions.

SHAP explanations describe the XGBoost model component and should not be interpreted as causal explanations or exact explanations of the final calibrated ensemble probability.

## Repository Structure

```text
Credit_Scoring_Capstone/
├── deployment/        # Flask API and model-serving code
├── frontend/          # React/Vite frontend
├── models/            # Saved trained model artifacts
├── notebooks/         # Modelling and analysis notebooks
├── outputs/           # Experimental results and figures
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md

This README also makes the distinction we've just spent time learning: **WoE LR is a benchmark, while regular LR + XGB + MLP form the stack**.

After saving it, stage the new README:

**TERMINAL — project root**
```powershell
git add README.md
git status