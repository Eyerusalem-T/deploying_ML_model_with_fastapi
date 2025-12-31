import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("🎯 Training Loan Approval Models...")

# Create training data
np.random.seed(42)
n_samples = 2000

# Generate features
data = pd.DataFrame({
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.randint(20000, 150000, n_samples),
    'loan_amount': np.random.randint(5000, 100000, n_samples)
})

# Calculate loan-to-income ratio
data['loan_to_income'] = data['loan_amount'] / data['income']

# Create labels based on business rules
data['approved'] = (
    (data['income'] > 40000) & 
    (data['loan_to_income'] < 0.5) & 
    (data['age'] > 25)
).astype(int)

# Add some noise (10%)
noise = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
data['approved'] = data['approved'] ^ noise

print(f"Dataset created: {data.shape}")
print(f"Approval rate: {data['approved'].mean():.1%}")

# Prepare features
X = data[['age', 'income', 'loan_amount', 'loan_to_income']]
y = data['approved']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
lr_model = LogisticRegression(random_state=42, max_iter=1000)

dt_model.fit(X_scaled, y)
lr_model.fit(X_scaled, y)

# Create models directory
os.makedirs('models', exist_ok=True)

# Save models
joblib.dump(dt_model, 'models/dt_model.joblib')
joblib.dump(lr_model, 'models/lr_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')

print("✅ Models saved to 'models/' directory")

# Test predictions
print("\n🎯 Test Examples:")
print("-" * 40)

test_cases = [
    ("Good applicant", 35, 65000, 20000),
    ("Risky applicant", 22, 25000, 50000),
    ("Excellent applicant", 45, 80000, 15000),
]

for desc, age, income, loan in test_cases:
    loan_to_income = loan / income
    features = [[age, income, loan, loan_to_income]]
    features_scaled = scaler.transform(features)
    
    dt_pred = dt_model.predict(features_scaled)[0]
    dt_prob = dt_model.predict_proba(features_scaled)[0][1]
    
    lr_pred = lr_model.predict(features_scaled)[0]
    lr_prob = lr_model.predict_proba(features_scaled)[0][1]
    
    print(f"\n{desc}:")
    print(f"  Age: {age}, Income: ${income:,}, Loan: ${loan:,}")
    print(f"  DT: {'✅ APPROVED' if dt_pred == 1 else '❌ REJECTED'} ({dt_prob:.1%})")
    print(f"  LR: {'✅ APPROVED' if lr_pred == 1 else '❌ REJECTED'} ({lr_prob:.1%})")

print("\n" + "="*50)
print(" Next: Run 'python api.py' to start the server")
print(" Then open 'frontend.html' in your browser")