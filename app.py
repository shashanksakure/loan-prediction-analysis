import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load XGBoost Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# HTML Template with Embedded CSS Styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Eligibility Predictor</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --success: #22c55e;
            --danger: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 850px;
            padding: 30px;
        }

        header {
            text-align: center;
            margin-bottom: 25px;
        }

        header h1 {
            font-size: 1.8rem;
            color: var(--accent);
            margin-bottom: 6px;
        }

        header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 16px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-muted);
        }

        input, select {
            background-color: var(--bg-color);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 10px 12px;
            border-radius: 6px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s ease;
        }

        input:focus, select:focus {
            border-color: var(--accent);
        }

        .btn-container {
            grid-column: 1 / -1;
            margin-top: 10px;
        }

        button {
            width: 100%;
            background-color: var(--accent);
            color: #000;
            font-weight: bold;
            padding: 12px;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.2s ease;
        }

        button:hover {
            background-color: var(--accent-hover);
            color: #fff;
        }

        #result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.1rem;
            font-weight: 600;
            display: none;
        }

        .approved {
            background-color: rgba(34, 197, 94, 0.15);
            border: 1px solid var(--success);
            color: var(--success);
        }

        .rejected {
            background-color: rgba(239, 68, 68, 0.15);
            border: 1px solid var(--danger);
            color: var(--danger);
        }
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>Loan Approval Prediction</h1>
        <p>Enter applicant details to verify loan eligibility</p>
    </header>

    <form action="/predict" method="POST">
        <div class="form-group">
            <label for="loan_id">Loan ID</label>
            <input type="number" id="loan_id" name="loan_id" value="1" required>
        </div>

        <div class="form-group">
            <label for="no_of_dependents">No. of Dependents</label>
            <input type="number" id="no_of_dependents" name="no_of_dependents" min="0" value="2" required>
        </div>

        <div class="form-group">
            <label for="education">Education Level</label>
            <select id="education" name="education" required>
                <option value="1">Graduate</option>
                <option value="0">Not Graduate</option>
            </select>
        </div>

        <div class="form-group">
            <label for="self_employed">Employment Status</label>
            <select id="self_employed" name="self_employed" required>
                <option value="0">Employed / Salaried</option>
                <option value="1">Self Employed</option>
            </select>
        </div>

        <div class="form-group">
            <label for="income_annum">Annual Income ($)</label>
            <input type="number" id="income_annum" name="income_annum" step="1000" value="5000000" required>
        </div>

        <div class="form-group">
            <label for="loan_amount">Loan Amount Requested ($)</label>
            <input type="number" id="loan_amount" name="loan_amount" step="1000" value="15000000" required>
        </div>

        <div class="form-group">
            <label for="loan_term">Loan Term (Years)</label>
            <input type="number" id="loan_term" name="loan_term" min="1" max="30" value="10" required>
        </div>

        <div class="form-group">
            <label for="cibil_score">CIBIL / Credit Score</label>
            <input type="number" id="cibil_score" name="cibil_score" min="300" max="900" value="750" required>
        </div>

        <div class="form-group">
            <label for="residential_assets_value">Residential Assets Value ($)</label>
            <input type="number" id="residential_assets_value" name="residential_assets_value" value="2000000" required>
        </div>

        <div class="form-group">
            <label for="commercial_assets_value">Commercial Assets Value ($)</label>
            <input type="number" id="commercial_assets_value" name="commercial_assets_value" value="1000000" required>
        </div>

        <div class="form-group">
            <label for="luxury_assets_value">Luxury Assets Value ($)</label>
            <input type="number" id="luxury_assets_value" name="luxury_assets_value" value="5000000" required>
        </div>

        <div class="form-group">
            <label for="bank_asset_value">Bank Asset Value ($)</label>
            <input type="number" id="bank_asset_value" name="bank_asset_value" value="3000000" required>
        </div>

        <div class="btn-container">
            <button type="submit">Evaluate Application</button>
        </div>
    </form>

    {% if prediction_text %}
        <div id="result" class="{{ result_class }}" style="display: block;">
            {{ prediction_text }}
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE,
            prediction_text="Model pickle file not found or corrupted.",
            result_class="rejected"
        )

    try:
        # Features order matching the training attributes:
        features = [
            float(request.form.get("loan_id", 0)),
            float(request.form.get("no_of_dependents", 0)),
            float(request.form.get("education", 0)),
            float(request.form.get("self_employed", 0)),
            float(request.form.get("income_annum", 0)),
            float(request.form.get("loan_amount", 0)),
            float(request.form.get("loan_term", 0)),
            float(request.form.get("cibil_score", 0)),
            float(request.form.get("residential_assets_value", 0)),
            float(request.form.get("commercial_assets_value", 0)),
            float(request.form.get("luxury_assets_value", 0)),
            float(request.form.get("bank_asset_value", 0))
        ]

        # Reshape input to 2D numpy array
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result_text = "Loan Approved!"
            status_class = "approved"
        else:
            result_text = "Loan Rejected."
            status_class = "rejected"

        return render_template_string(
            HTML_TEMPLATE,
            prediction_text=f"Prediction Result: {result_text}",
            result_class=status_class
        )

    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            prediction_text=f"Error processing request: {str(e)}",
            result_class="rejected"
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
