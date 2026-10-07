# System Architecture

## 1. Overview

The completed project integrates the final feature-engineered credit-scoring pipeline with a Flask REST API, React dashboard, and Docker-based runtime.

The high-level architecture is:

```text
Browser
   │
   ▼
React Dashboard
   │
   │ JSON
   ▼
Flask REST API
   │
   ▼
model_service.py
   │
   ├── Logistic Regression
   │   Ratios + Logs
   │
   ├── XGBoost
   │   Ratios
   │
   └── PyTorch MLP
       Ratios + Logs
          │
          ▼
   Meta Logistic Regression
          │
          ▼
       Raw Stack PD
          │
          ▼
    Isotonic Calibration
          │
          ▼
       Final PD
        /    \
       /      \
      ▼        ▼
Financial    XGBoost
Engine       SHAP
      \        /
       \      /
        ▼    ▼
     JSON Response
          │
          ▼
    React Dashboard
```

## 2. Backend

The backend is located in:

```text
deployment/
├── app.py
├── model_service.py
├── requirements.txt
├── sample_applicant.json
└── Dockerfile
```

### `model_service.py`

`model_service.py` is responsible for loading the frozen final model artifacts and executing the prediction pipeline.

The service performs:

1. applicant-data preparation;
2. final feature engineering;
3. model-specific preprocessing;
4. Logistic Regression inference;
5. XGBoost inference;
6. MLP inference;
7. meta-model inference;
8. isotonic calibration;
9. XGBoost SHAP explanation.

The main prediction path is:

```text
Applicant Record
       │
       ▼
make_features(...)
       │
       ├── financial ratios
       └── log-transformed features
       │
       ▼
Model-Specific Preprocessing
       │
       ├── LR Preprocessor
       ├── XGB Preprocessor
       └── MLP Preprocessor
       │
       ▼
Base Model Probabilities
       │
       ▼
[LR PD, XGB PD, MLP PD]
       │
       ▼
Meta Logistic Regression
       │
       ▼
Raw Ensemble PD
       │
       ▼
Isotonic Calibration
       │
       ▼
Final Calibrated PD
```

The prediction service therefore produces:

- Logistic Regression PD;
- XGBoost PD;
- MLP PD;
- raw ensemble PD;
- final calibrated PD.

It also provides:

- Expected Loss;
- indicative interest rate;
- XGBoost SHAP explanation.

## 3. Flask API

`deployment/app.py` exposes the HTTP interface.

The main endpoints are:

```text
GET /health
POST /predict
```

A prediction request follows the general structure:

```json
{
  "applicant": {
    "AMT_CREDIT": 500000,
    "...": "complete applicant feature record"
  },
  "lgd": 0.40
}
```

The backend expects a complete applicant feature record.

The dashboard exposes only a selected subset of variables for editing. Applicant variables that are not edited remain at their baseline values.

The API response contains the model probabilities, financial outputs, and explanation information required by the frontend.

## 4. Final Saved Model Artifacts

The frozen final deployment artifacts are stored in:

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

These artifacts correspond to the final feature-engineered pipeline.

### Logistic Regression

The Logistic Regression deployment uses:

```text
lr_preprocessor.joblib
lr_model.joblib
```

Its final feature configuration includes financial ratios and retained log-transformed financial features.

### XGBoost

The XGBoost deployment uses:

```text
xgb_preprocessor.joblib
xgb_model.joblib
```

Its final feature configuration includes the retained financial-ratio features.

XGBoost is also the component used for SHAP explanations.

### MLP

The MLP deployment uses:

```text
mlp_preprocessor.joblib
mlp_state.pt
mlp_metadata.json
```

The metadata file stores the information required to reconstruct the deployed network architecture.

Its final feature configuration includes financial ratios and retained log-transformed features.

### Meta-Learner

The stacking layer uses:

```text
meta_logistic.joblib
```

It receives:

```text
[LR PD, XGB PD, MLP PD]
```

and returns the raw ensemble probability.

### Calibration

The final probability mapping uses:

```text
isotonic_calibrator.joblib
```

The isotonic calibrator transforms the raw ensemble probability into the deployed final PD.

### Feature Configuration

The file:

```text
feature_config.json
```

stores configuration information required by the final feature-engineering process.

This helps keep the deployed transformations aligned with the final modelling experiment.

## 5. Historical and Final Model Directories

The repository may contain both:

```text
models/
models_final/
```

They have different roles.

`models/` contains artifacts associated with earlier stages of the experimental workflow.

`models_final/` contains the frozen feature-engineered artifacts used by the completed deployment.

The Dockerized final application uses:

```text
models_final/
```

The earlier `models/` directory should therefore not be interpreted as the currently deployed final model.

## 6. Frontend

The React application is located in:

```text
frontend/
```

The dashboard operates on a complete baseline applicant record.

A selected subset of meaningful variables is exposed for interactive scenario analysis.

Editable numerical variables include examples such as:

- income;
- credit amount;
- annuity;
- goods price;
- age;
- employment duration;
- external-source scores.

Editable categorical variables include examples such as:

- gender;
- car ownership;
- property ownership;
- education;
- income type;
- housing type.

Variables not exposed in the interface retain their baseline applicant values.

## 7. Dashboard Outputs

The interface displays:

- baseline-versus-current-scenario comparison;
- Logistic Regression probability;
- XGBoost probability;
- MLP probability;
- raw ensemble probability;
- calibrated final PD;
- Expected Loss;
- indicative interest rate;
- XGBoost SHAP drivers.

The interface is designed as a decision-support and scenario-analysis prototype rather than an automatic lending-decision system.

## 8. Financial Scenario Layer

The financial layer uses:

```text
Expected Loss = PD × LGD × EAD
```

where:

```text
PD  = final calibrated model probability
LGD = user-selected scenario assumption
EAD = AMT_CREDIT proxy
```

The indicative-rate prototype uses:

```text
Indicative Rate = 0.09 + (PD × LGD)
```

LGD is handled outside the predictive model.

Therefore:

```text
Applicant Features
       │
       ▼
      PD
       │
       ├──────────────┐
       │              │
       ▼              ▼
     × LGD          × LGD
       │              │
       ▼              ▼
Expected Loss    Indicative Rate
```

Changing LGD changes the financial outputs but does not cause the predictive model to generate a different PD.

## 9. Explainability Layer

SHAP explanations are generated for the XGBoost component.

The general explanation path is:

```text
Applicant
   │
   ▼
XGBoost Feature Engineering
   │
   ▼
XGBoost Preprocessor
   │
   ▼
XGBoost Model
   │
   ▼
Tree SHAP
   │
   ▼
Top Local Drivers
```

The SHAP values explain XGBoost behaviour.

They are not an exact decomposition of:

- the Logistic Regression model;
- the MLP;
- the meta-logistic model;
- isotonic calibration; or
- the complete final PD.

They should also not be interpreted as causal effects.

## 10. Docker Architecture

The application uses two containers managed by Docker Compose.

```text
                 Docker Compose
                /              \
               /                \
              ▼                  ▼
       Backend Container    Frontend Container
        Python / Flask       React → Nginx
              │                  │
              │                  │
           Port 5000          Port 80
              │                  │
              ▼                  ▼
       localhost:5000      localhost:8080
```

### Backend Container

The backend container:

1. uses Python 3.12 slim;
2. installs the Python deployment dependencies;
3. installs CPU-only PyTorch;
4. copies the deployment code;
5. copies `models_final/`;
6. starts Flask on port 5000.

The relevant final model copy step is conceptually:

```text
Repository models_final/
          ↓
Container /app/models_final/
```

Host mapping:

```text
5000:5000
```

### Frontend Container

The frontend uses a multi-stage Docker build.

The first stage uses Node to build the React/Vite application.

The second stage uses Nginx to serve the generated production files.

Host mapping:

```text
8080:80
```

The user therefore accesses the dashboard through:

```text
http://localhost:8080
```

## 11. Request Flow

A normal prediction request follows this sequence:

```text
1. User changes applicant variables in React
                    ↓
2. React constructs complete applicant scenario
                    ↓
3. POST /predict
                    ↓
4. Flask validates request
                    ↓
5. model_service performs feature engineering
                    ↓
6. LR, XGB and MLP generate probabilities
                    ↓
7. MetaLR produces raw ensemble probability
                    ↓
8. Isotonic mapping produces final PD
                    ↓
9. Financial outputs are calculated
                    ↓
10. XGBoost SHAP explanation is generated
                    ↓
11. JSON response returned to React
                    ↓
12. Dashboard displays scenario results
```

## 12. Running the Complete Application

From the repository root:

```bash
docker compose up --build
```

After startup:

```text
Frontend:
http://localhost:8080

Backend health:
http://localhost:5000/health
```

To stop the application:

```bash
docker compose down
```

After the images have already been built, the application can normally be restarted using:

```bash
docker compose up
```

## 13. Development and Local Production-Style Workflows

During development, Flask and React can be run independently to support faster iteration.

The completed local production-style configuration uses Docker Compose to run the frontend and backend together.

The final architecture demonstrates:

```text
Model Artifacts
      +
REST Model Serving
      +
Interactive Frontend
      +
Containerized Runtime
```

without requiring the original training dataset simply to run inference.

## 14. Deployment Scope

The implemented architecture demonstrates:

- model serialization;
- reproducible model-specific preprocessing;
- feature-engineered inference;
- multi-model prediction;
- OOF-derived stacking architecture;
- probability calibration;
- Expected Loss calculation;
- indicative-rate scenario analysis;
- local XGBoost SHAP explanation;
- REST-based model serving;
- interactive frontend integration;
- Dockerized execution.

## 15. Production Limitations

The current architecture is a local academic research prototype.

It does not implement:

- live credit-bureau integration;
- live banking/payment-system integration;
- user authentication and authorization;
- public HTTPS deployment;
- production secrets management;
- production database persistence;
- automated model monitoring;
- drift monitoring;
- CI/CD;
- formal model-governance workflows;
- production fairness controls;
- regulatory lending-decision infrastructure.

The application should therefore be interpreted as evidence that the modelling framework can be integrated into a functioning decision-support prototype, rather than as a production-ready lending platform.