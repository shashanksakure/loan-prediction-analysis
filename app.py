import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load XGBoost Model
MODEL_PATH = "model.pkl"

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# Features expected by model.pkl
FEATURE_NAMES = [
    "loan_id",
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]

# Integrated HTML Template with embedded CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Approval Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --input-bg: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --border-color: #475569;
            --success-color: #10b981;
            --danger-color: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background-color: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
            padding: 2.5rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group label {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            text-transform: capitalize;
        }

        .form-group input, .form-group select {
            background-color: var(--input-bg);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .form-group input:focus, .form-group select:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background-color: var(--accent-color);
            color: white;
            border: none;
            padding: 0.9rem 1.5rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
        }

        .submit-btn:hover {
            background-color: var(--accent-hover);
        }

        .submit-btn:active {
            transform: scale(0.99);
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            display: none;
            animation: fadeIn 0.3s ease-in-out forwards;
        }

        .result-card.success {
            background-color: rgba(16, 185, 129, 0.15);
            border: 1px solid var(--success-color);
            color: #34d399;
        }

        .result-card.danger {
            background-color: rgba(239, 68, 68, 0.15);
            border: 1px solid var(--danger-color);
            color: #f87171;
        }

        .result-card h2 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }

        .result-card p {
            font-size: 1rem;
            color: var(--text-primary);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <div class="container">
        <div class="header">
            <h1>Loan Eligibility Assessment</h1>
            <p>Enter applicant details below to make an inference via XGBoost</p>
        </div>

        <form id="predictionForm" class="form-grid">
            <div class="form-group">
                <label for="loan_id">Loan ID</label>
                <input type="number" id="loan_id" name="loan_id" value="101" required>
            </div>
            
            <div class="form-group">
                <label for="no_of_dependents">Number of Dependents</label>
                <input type="number" id="no_of_dependents" name="no_of_dependents" value="2" min="0" required>
            </div>

            <div class="form-group">
                <label for="education">Education</label>
                <select id="education" name="education" required>
                    <option value="1">Graduate (1)</option>
                    <option value="0">Not Graduate (0)</option>
                </select>
            </div>

            <div class="form-group">
                <label for="self_employed">Self Employed</label>
                <select id="self_employed" name="self_employed" required>
                    <option value="0">No (0)</option>
                    <option value="1">Yes (1)</option>
                </select>
            </div>

            <div class="form-group">
                <label for="income_annum">Annual Income ($)</label>
                <input type="number" id="income_annum" name="income_annum" value="9600000" required>
            </div>

            <div class="form-group">
                <label for="loan_amount">Loan Amount ($)</label>
                <input type="number" id="loan_amount" name="loan_amount" value="29900000" required>
            </div>

            <div class="form-group">
                <label for="loan_term">Loan Term (Years)</label>
                <input type="number" id="loan_term" name="loan_term" value="12" required>
            </div>

            <div class="form-group">
                <label for="cibil_score">CIBIL Score</label>
                <input type="number" id="cibil_score" name="cibil_score" value="778" min="300" max="900" required>
            </div>

            <div class="form-group">
                <label for="residential_assets_value">Residential Asset Value ($)</label>
                <input type="number" id="residential_assets_value" name="residential_assets_value" value="2400000" required>
            </div>

            <div class="form-group">
                <label for="commercial_assets_value">Commercial Asset Value ($)</label>
                <input type="number" id="commercial_assets_value" name="commercial_assets_value" value="17600000" required>
            </div>

            <div class="form-group">
                <label for="luxury_assets_value">Luxury Asset Value ($)</label>
                <input type="number" id="luxury_assets_value" name="luxury_assets_value" value="22700000" required>
            </div>

            <div class="form-group">
                <label for="bank_asset_value">Bank Asset Value ($)</label>
                <input type="number" id="bank_asset_value" name="bank_asset_value" value="8000000" required>
            </div>

            <button type="submit" class="submit-btn">Evaluate Application</button>
        </form>

        <div id="resultCard" class="result-card">
            <h2 id="resultTitle">Result</h2>
            <p id="resultDetails"></p>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((value, key) => data[key] = parseFloat(value));

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            const resultCard = document.getElementById('resultCard');
            const resultTitle = document.getElementById('resultTitle');
            const resultDetails = document.getElementById('resultDetails');

            if (result.status === 'success') {
                resultCard.style.display = 'block';
                resultCard.className = `result-card ${result.prediction === 1 ? 'success' : 'danger'}`;
                resultTitle.innerText = result.prediction === 1 ? 'Loan Approved' : 'Loan Rejected';
                resultDetails.innerText = `Probability: ${(result.probability * 100).toFixed(2)}%`;
            } else {
                resultCard.style.display = 'block';
                resultCard.className = 'result-card danger';
                resultTitle.innerText = 'Error';
                resultDetails.innerText = result.message;
            }
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"status": "error", "message": "Model pickle file not found on server."}), 500

    try:
        data = request.get_json()
        
        # Extract features in exact order
        features = [float(data[feature]) for feature in FEATURE_NAMES]
        input_array = np.array([features])

        # Inference
        prediction = model.predict(input_array)[0]
        
        # If probability estimation is available
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
