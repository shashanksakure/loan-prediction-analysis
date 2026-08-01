import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Absolute path targeting the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = None

# Attempt to load the model
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Successfully loaded model.pkl")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"CRITICAL ERROR: model.pkl not found at {MODEL_PATH}")

FEATURE_NAMES = [
    "loan_id", "no_of_dependents", "education", "self_employed", 
    "income_annum", "loan_amount", "loan_term", "cibil_score", 
    "residential_assets_value", "commercial_assets_value", 
    "luxury_assets_value", "bank_asset_value"
]

@app.route("/", methods=["GET"])
def home():
    return "XGBoost Loan Prediction API is Running."

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "status": "error", 
            "message": f"Model file missing or failed to load on server. Expected path: {MODEL_PATH}"
        }), 500

    try:
        data = request.get_json()
        features = [float(data[feature]) for feature in FEATURE_NAMES]
        input_array = np.array([features])

        prediction = model.predict(input_array)[0]
        
        probability = 0.0
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_array)[0][int(prediction)]

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
