import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("Churn_Modelling.csv")

# Drop unnecessary columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Convert categorical variables into numerical using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Select features and target
selected_features = ['CreditScore', 'Age', 'Balance', 'NumOfProducts']
X = data[selected_features]
y = data['Exited']

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "churn_model.pkl")  # Save the model
joblib.dump(scaler, "scaler.pkl")      # Save the scaler

print("✅ Model and scaler saved successfully as churn_model.pkl and scaler.pkl!")
