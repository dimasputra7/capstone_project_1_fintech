from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

from model_service import (
    predict_pd,
    calculate_financial_outputs,
    explain_xgb
)

app = Flask(__name__)
CORS(app)


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "message": "Credit scoring API is running."
    })


# --------------------------------------------------
# Credit Prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Receive JSON request
        request_data = request.get_json()

        if not request_data:
            return jsonify({
                "error": "No data provided."
            }), 400

        # Keep applicant features separate from financial assumptions
        applicant_data = request_data.get("applicant")
        lgd = request_data.get("lgd", 0.40)

        if not applicant_data:
            return jsonify({
                "error": "No applicant data provided."
            }), 400

        # Validate LGD
        lgd = float(lgd)

        if lgd < 0 or lgd > 1:
            return jsonify({
                "error": "LGD must be between 0 and 1."
            }), 400

        # Convert only applicant features to DataFrame
        applicant_df = pd.DataFrame([applicant_data])

        # Predict Probability of Default
        prediction = predict_pd(applicant_df)

        # Explain XGBoost component
        explanation = explain_xgb(applicant_df, top_n=5)

        # AMT_CREDIT is used as EAD proxy
        exposure = float(applicant_data["AMT_CREDIT"])

        # Calculate Expected Loss and Rate
        financial = calculate_financial_outputs(
            pd_value=prediction["final_pd"],
            exposure=exposure,
            lgd=lgd,
            fixed_rate=0.09
        )

        # Combine results
        result = {
            **prediction,
            **financial,
            "xgb_explanation": explanation
        }

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)