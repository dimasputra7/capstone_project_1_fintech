# System Architecture

## 1. Overview

The final project combines the trained credit-risk models with a Flask REST API, React dashboard, and Docker-based runtime.

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
   ├── XGBoost
   └── PyTorch MLP
          │
          ▼
   Meta Logistic Regression
          │
          ▼
    Isotonic Calibration
          │
          ▼
      Final PD
       /    \
      /      \
Financial    XGBoost
Engine       SHAP
      \      /
       \    /
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

The model service loads the saved preprocessing and model artifacts and provides the main prediction functions.

The prediction pipeline produces:

- Logistic Regression PD
- XGBoost PD
- MLP PD
- Raw ensemble PD
- Final calibrated PD

It also provides:

- Expected Loss calculation
- Indicative-rate calculation
- XGBoost SHAP explanation

### Flask API

`app.py` exposes the HTTP interface.

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

The backend expects a complete applicant record rather than only the fields visible in the dashboard.

The response contains component-model probabilities, final calibrated PD, financial outputs, and XGBoost explanation information.

## 3. Saved Model Artifacts

The deployed model artifacts are stored in:

```text
models/
├── baseline_lr.joblib
├── xgb_pipeline.joblib
├── mlp_preprocessor.joblib
├── mlp_metadata.joblib
├── mlp_state_dict.pt
├── meta_logistic.joblib
├── platt_calibrator.joblib
└── isotonic_calibrator.joblib
```

The final prediction path uses the baseline LR, XGBoost, MLP, meta-logistic model, and isotonic calibrator.

The Platt calibrator is retained as an experimental artifact from the calibration comparison.

## 4. Frontend

The React application is located in `frontend/`.

The dashboard operates on a complete baseline applicant record. A selected subset of meaningful variables is exposed to the user for scenario analysis.

Examples include:

- income
- credit amount
- annuity
- goods price
- age
- employment duration
- external source scores
- selected categorical applicant characteristics

Variables that are not exposed in the interface retain their baseline values.

The interface displays:

- component-model probabilities
- raw ensemble probability
- calibrated final PD
- Expected Loss
- indicative rate
- XGBoost SHAP drivers
- baseline-versus-current-scenario comparison

LGD is controlled separately because it is a financial scenario assumption rather than a predictive model feature.

## 5. Docker Architecture

The application uses two containers managed by Docker Compose.

### Backend Container

The backend container:

1. Uses Python 3.12 slim.
2. Installs the Python deployment requirements.
3. Installs CPU-only PyTorch.
4. Copies the deployment code.
5. Copies the saved model artifacts.
6. Starts Flask on port 5000.

Host mapping:

```text
5000:5000
```

### Frontend Container

The frontend uses a multi-stage build:

1. Node builds the React/Vite application.
2. Nginx serves the resulting production files.

Host mapping:

```text
8080:80
```

## 6. Running the Complete Application

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

After the images have been built, the application can normally be restarted with:

```bash
docker compose up
```

## 7. Development and Production-Style Workflows

During development, Flask can run in Docker while React runs through the Vite development server for faster frontend iteration.

The final local production-style configuration runs both services using Docker Compose.

The current frontend/backend configuration is designed for local execution. Public cloud deployment would require additional production configuration and is outside the implemented project scope.

## 8. Deployment Scope

The implemented system demonstrates:

- model serialization;
- reproducible preprocessing;
- multi-model inference;
- probability calibration;
- financial calculations;
- model explanation;
- REST-based model serving;
- interactive frontend integration;
- containerized deployment.

It should be treated as a research prototype rather than a production lending platform.