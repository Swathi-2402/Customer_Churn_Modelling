from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load dataset for feature scaling
data = pd.read_csv("Churn_Modelling.csv")
selected_features = ['CreditScore', 'Age', 'Balance', 'NumOfProducts']
X = data[selected_features]

# Load trained model and scaler
try:
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading model/scaler: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Extract input values
        try:
            credit_score = float(data.get("credit_score", 0))
            age = int(data.get("age", 0))
            balance = float(data.get("balance", 0))
            num_products = int(data.get("num_products", 0))
        except ValueError:
            return jsonify({"error": "Invalid input types"}), 400

        # Prepare input for prediction
        input_data = np.array([[credit_score, age, balance, num_products]])
        input_scaled = scaler.transform(input_data)  # Scale input

        # Predict churn
        prediction = model.predict(input_scaled)[0]

        # Format the result
        result = "🔴 Churn (Customer will Leave)" if prediction == 1 else "🟢 Stay (Customer will Remain)"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
