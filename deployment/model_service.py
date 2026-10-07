import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import shap

from pathlib import Path


# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# New feature-engineered final models
MODEL_DIR = BASE_DIR / "models_final"


# ==================================================
# FEATURE ENGINEERING
# ==================================================

RATIO_DEFINITIONS = {
    "CREDIT_INCOME_RATIO": (
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL"
    ),
    "ANNUITY_INCOME_RATIO": (
        "AMT_ANNUITY",
        "AMT_INCOME_TOTAL"
    ),
    "CREDIT_GOODS_RATIO": (
        "AMT_CREDIT",
        "AMT_GOODS_PRICE"
    ),
    "ANNUITY_CREDIT_RATIO": (
        "AMT_ANNUITY",
        "AMT_CREDIT"
    ),
}

LOG_COLUMNS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE"
]


def safe_ratio(numerator, denominator):
    denominator = denominator.replace(0, np.nan)

    result = numerator / denominator

    return result.replace(
        [np.inf, -np.inf],
        np.nan
    )


def make_features(applicant_df, representation):
    """
    Create the same engineered feature representation
    used during final model training.

    representation:
        "ratios"
        "ratios_logs"
    """

    data = applicant_df.copy()

    # ------------------------------
    # Financial ratios
    # ------------------------------

    if representation in [
        "ratios",
        "ratios_logs"
    ]:

        for name, (num, den) in RATIO_DEFINITIONS.items():

            data[name] = safe_ratio(
                data[num],
                data[den]
            )

    # ------------------------------
    # Log monetary variables
    # ------------------------------

    if representation == "ratios_logs":

        for col in LOG_COLUMNS:

            data[f"LOG_{col}"] = np.log1p(
                data[col].clip(lower=0)
            )

    return data


# ==================================================
# MLP ARCHITECTURE
# ==================================================

class CreditMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ==================================================
# LOAD NEW TRAINED ARTIFACTS
# ==================================================

# Logistic Regression
lr_preprocessor = joblib.load(
    MODEL_DIR / "lr_preprocessor.joblib"
)

lr_model = joblib.load(
    MODEL_DIR / "lr_model.joblib"
)


# XGBoost
xgb_preprocessor = joblib.load(
    MODEL_DIR / "xgb_preprocessor.joblib"
)

xgb_model = joblib.load(
    MODEL_DIR / "xgb_model.joblib"
)


# MLP
mlp_preprocessor = joblib.load(
    MODEL_DIR / "mlp_preprocessor.joblib"
)

with open(
    MODEL_DIR / "mlp_metadata.json",
    "r",
    encoding="utf-8"
) as f:
    mlp_metadata = json.load(f)


mlp_model = CreditMLP(
    input_dim=mlp_metadata["input_dim"]
)

mlp_model.load_state_dict(
    torch.load(
        MODEL_DIR / "mlp_state.pt",
        map_location="cpu"
    )
)

mlp_model.eval()


# Meta Logistic Regression
meta_model = joblib.load(
    MODEL_DIR / "meta_logistic.joblib"
)


# Isotonic calibration
isotonic_calibrator = joblib.load(
    MODEL_DIR / "isotonic_calibrator.joblib"
)


# ==================================================
# SHAP SETUP
# ==================================================

xgb_feature_names = (
    xgb_preprocessor
    .get_feature_names_out()
)

xgb_feature_names = [
    name
    .replace("num__", "")
    .replace("cat__", "")
    for name in xgb_feature_names
]

shap_explainer = shap.TreeExplainer(
    xgb_model
)


# ==================================================
# PD PREDICTION
# ==================================================

def predict_pd(applicant_df):

    # --------------------------------------------------
    # Logistic Regression
    # Uses ratios + logarithmic monetary features
    # --------------------------------------------------

    lr_features = make_features(
        applicant_df,
        "ratios_logs"
    )

    lr_input = lr_preprocessor.transform(
        lr_features
    )

    lr_prob = lr_model.predict_proba(
        lr_input
    )[:, 1]


    # --------------------------------------------------
    # XGBoost
    # Uses financial ratios
    # --------------------------------------------------

    xgb_features = make_features(
        applicant_df,
        "ratios"
    )

    xgb_input = xgb_preprocessor.transform(
        xgb_features
    )

    xgb_prob = xgb_model.predict_proba(
        xgb_input
    )[:, 1]


    # --------------------------------------------------
    # MLP
    # Uses ratios + logarithmic monetary features
    # --------------------------------------------------

    mlp_features = make_features(
        applicant_df,
        "ratios_logs"
    )

    mlp_input = mlp_preprocessor.transform(
        mlp_features
    )

    mlp_input = np.asarray(
        mlp_input,
        dtype=np.float32
    )

    mlp_tensor = torch.tensor(
        mlp_input,
        dtype=torch.float32
    )

    with torch.no_grad():

        mlp_logits = mlp_model(
            mlp_tensor
        )

        mlp_prob = torch.sigmoid(
            mlp_logits
        ).cpu().numpy().ravel()


    # --------------------------------------------------
    # Stacking
    # --------------------------------------------------

    meta_features = np.column_stack([
        lr_prob,
        xgb_prob,
        mlp_prob
    ])

    raw_pd = meta_model.predict_proba(
        meta_features
    )[:, 1]


    # --------------------------------------------------
    # Isotonic calibration
    # --------------------------------------------------

    final_pd = isotonic_calibrator.predict(
        raw_pd
    )


    return {
        "lr_pd": float(lr_prob[0]),
        "xgb_pd": float(xgb_prob[0]),
        "mlp_pd": float(mlp_prob[0]),
        "raw_ensemble_pd": float(raw_pd[0]),
        "final_pd": float(final_pd[0])
    }


# ==================================================
# XGBOOST SHAP EXPLANATION
# ==================================================

def explain_xgb(applicant_df, top_n=5):
    """
    Explain the XGBoost component of the final ensemble.

    XGBoost uses the financial-ratio feature representation.
    """

    # Same feature engineering used during XGB training
    xgb_features = make_features(
        applicant_df,
        "ratios"
    )

    transformed = xgb_preprocessor.transform(
        xgb_features
    )

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    shap_values = shap_explainer(
        transformed
    )

    values = shap_values.values[0]

    explanation_df = pd.DataFrame({
        "feature": xgb_feature_names,
        "shap_value": values
    })

    explanation_df["absolute_shap"] = (
        explanation_df["shap_value"].abs()
    )

    explanation_df = (
        explanation_df
        .sort_values(
            "absolute_shap",
            ascending=False
        )
        .head(top_n)
    )

    results = []

    for _, row in explanation_df.iterrows():

        results.append({
            "feature": str(row["feature"]),

            "shap_value": float(
                row["shap_value"]
            ),

            "direction": (
                "Higher model output"
                if row["shap_value"] > 0
                else "Lower model output"
            )
        })

    return results


# ==================================================
# FINANCIAL CALCULATIONS
# ==================================================

def calculate_financial_outputs(
    pd_value,
    exposure,
    lgd=0.40,
    fixed_rate=0.09
):

    expected_loss = (
        pd_value
        * lgd
        * exposure
    )

    indicative_rate = (
        fixed_rate
        + pd_value * lgd
    )

    return {
        "ead_proxy": float(exposure),
        "lgd": float(lgd),
        "expected_loss": float(expected_loss),
        "indicative_rate": float(indicative_rate)
    }