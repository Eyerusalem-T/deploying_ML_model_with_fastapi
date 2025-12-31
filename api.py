"""
FastAPI Backend for Loan Prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
from typing import Dict, Any

print("🚀 Starting Loan Prediction API...")

# Load models
try:
    dt_model = joblib.load('models/dt_model.joblib')
    lr_model = joblib.load('models/lr_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("💡 Run 'python train.py' first!")
    exit()

# Create FastAPI app
app = FastAPI(
    title="Loan Approval API",
    description="API for predicting loan approval",
    version="1.0"
)

# Enable CORS - VERY IMPORTANT for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define request model
class LoanRequest(BaseModel):
    age: int
    income: float
    loan_amount: float

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Loan Approval Prediction API",
        "status": "running",
        "endpoints": {
            "GET /": "API info",
            "GET /health": "Health check",
            "POST /predict": "Get predictions from both models"
        },
        "example_request": {
            "age": 35,
            "income": 65000,
            "loan_amount": 25000
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "server": "running"
    }

# Main prediction endpoint
@app.post("/predict")
async def predict(request: LoanRequest) -> Dict[str, Any]:
    """
    Predict loan approval for given applicant data
    """
    try:
        # Calculate loan-to-income ratio
        loan_to_income = request.loan_amount / request.income
        
        # Prepare features array
        features = np.array([[
            float(request.age),
            float(request.income),
            float(request.loan_amount),
            float(loan_to_income)
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get predictions from both models
        dt_pred = dt_model.predict(features_scaled)[0]
        dt_proba = dt_model.predict_proba(features_scaled)[0]
        
        lr_pred = lr_model.predict(features_scaled)[0]
        lr_proba = lr_model.predict_proba(features_scaled)[0]
        
        # Calculate risk factors
        risk_factors = []
        if request.age < 25:
            risk_factors.append(f"Young applicant (age {request.age})")
        if request.income < 30000:
            risk_factors.append(f"Low income (${request.income:,.0f})")
        if loan_to_income > 0.5:
            risk_factors.append(f"High loan-to-income ratio ({loan_to_income:.2f})")
        
        # Prepare response
        response = {
            "success": True,
            "input_data": {
                "age": request.age,
                "income": request.income,
                "loan_amount": request.loan_amount,
                "loan_to_income_ratio": round(loan_to_income, 3)
            },
            "predictions": {
                "decision_tree": {
                    "prediction": "APPROVED" if dt_pred == 1 else "REJECTED",
                    "approved": bool(dt_pred == 1),
                    "probability": round(float(dt_proba[1]), 3),
                    "percentage": f"{dt_proba[1] * 100:.1f}%",
                    "confidence": "HIGH" if max(dt_proba) > 0.8 else "MEDIUM" if max(dt_proba) > 0.6 else "LOW"
                },
                "logistic_regression": {
                    "prediction": "APPROVED" if lr_pred == 1 else "REJECTED",
                    "approved": bool(lr_pred == 1),
                    "probability": round(float(lr_proba[1]), 3),
                    "percentage": f"{lr_proba[1] * 100:.1f}%",
                    "confidence": "HIGH" if max(lr_proba) > 0.8 else "MEDIUM" if max(lr_proba) > 0.6 else "LOW"
                }
            },
            "analysis": {
                "risk_factors": risk_factors,
                "loan_to_income_ratio": round(loan_to_income, 3),
                "monthly_payment": round(request.loan_amount / 36, 2),
                "monthly_income": round(request.income / 12, 2),
                "payment_burden": round((request.loan_amount / 36) / (request.income / 12), 3)
            },
            "summary": {
                "models_agree": dt_pred == lr_pred,
                "final_decision": "APPROVED" if dt_pred == 1 and lr_pred == 1 else "REJECTED" if dt_pred == 0 and lr_pred == 0 else "REVIEW NEEDED",
                "average_approval_probability": round((dt_proba[1] + lr_proba[1]) / 2, 3)
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🌐 API Server Starting...")
    print("📡 Endpoint: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🎮 Open 'frontend.html' in browser to use the app")
    print("="*50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)