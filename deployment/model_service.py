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
MODEL_DIR = BASE_DIR / "models"


# ==================================================
# MLP ARCHITECTURE
# ==================================================

class CreditMLP(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


# ==================================================
# LOAD TRAINED MODELS
# ==================================================

baseline_lr = joblib.load(
    MODEL_DIR / "baseline_lr.joblib"
)

xgb_pipeline = joblib.load(
    MODEL_DIR / "xgb_pipeline.joblib"
)

meta_model = joblib.load(
    MODEL_DIR / "meta_logistic.joblib"
)

isotonic_calibrator = joblib.load(
    MODEL_DIR / "isotonic_calibrator.joblib"
)

mlp_preprocessor = joblib.load(
    MODEL_DIR / "mlp_preprocessor.joblib"
)

mlp_metadata = joblib.load(
    MODEL_DIR / "mlp_metadata.joblib"
)


mlp_model = CreditMLP(
    input_size=mlp_metadata["input_size"]
)

mlp_model.load_state_dict(
    torch.load(
        MODEL_DIR / "mlp_state_dict.pt",
        map_location="cpu"
    )
)

mlp_model.eval()


# ==================================================
# PD PREDICTION
# ==================================================

def predict_pd(applicant_df):

    # Logistic Regression
    lr_prob = baseline_lr.predict_proba(
        applicant_df
    )[:, 1]

    # XGBoost
    xgb_prob = xgb_pipeline.predict_proba(
        applicant_df
    )[:, 1]

    # MLP preprocessing
    mlp_input = mlp_preprocessor.transform(
        applicant_df
    )

    mlp_input = np.asarray(
        mlp_input,
        dtype=np.float32
    )

    mlp_tensor = torch.tensor(
        mlp_input,
        dtype=torch.float32
    )

    # MLP prediction
    with torch.no_grad():

        mlp_logits = mlp_model(
            mlp_tensor
        )

        mlp_prob = torch.sigmoid(
            mlp_logits
        ).cpu().numpy().ravel()

    # Meta-model inputs
    meta_features = np.column_stack([
        lr_prob,
        xgb_prob,
        mlp_prob
    ])

    # Raw ensemble PD
    raw_pd = meta_model.predict_proba(
        meta_features
    )[:, 1]

    # Final calibrated PD
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

def explain_xgb(applicant_df, top_n=5):
    """
    Return the largest local SHAP contributions for the
    XGBoost component of the ensemble.
    """

    # Apply exactly the same preprocessing used by XGBoost
    transformed = xgb_preprocessor.transform(applicant_df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    # Calculate SHAP values
    shap_values = shap_explainer(transformed)

    values = shap_values.values[0]

    explanation_df = pd.DataFrame({
        "feature": xgb_feature_names,
        "shap_value": values
    })

    # Rank by magnitude, while retaining the sign
    explanation_df["absolute_shap"] = (
        explanation_df["shap_value"].abs()
    )

    explanation_df = (
        explanation_df
        .sort_values("absolute_shap", ascending=False)
        .head(top_n)
    )

    results = []

    for _, row in explanation_df.iterrows():
        results.append({
            "feature": str(row["feature"]),
            "shap_value": float(row["shap_value"]),
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

# XGBoost components used for local SHAP explanations
xgb_preprocessor = xgb_pipeline.named_steps["preprocessor"]
xgb_model = xgb_pipeline.named_steps["model"]

xgb_feature_names = xgb_preprocessor.get_feature_names_out()

xgb_feature_names = [
    name.replace("num__", "").replace("cat__", "")
    for name in xgb_feature_names
]

shap_explainer = shap.TreeExplainer(xgb_model)