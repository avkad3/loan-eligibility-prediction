from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("tuned_random_forest.pkl")

app = FastAPI(
    title="Loan Eligibility Prediction API",
    description="API for predicting loan eligibility using a trained Random Forest model",
    version="1.0.0"
)

class LoanApplication(BaseModel):
    credit_score: float
    annual_income: float
    loan_amount: float
    loan_term_months: float
    employment_status: str
    property_ownership: str
    debt_to_income_ratio: float
    num_dependents: float
    monthly_income: float
    estimated_monthly_payment: float
    total_dti: float

@app.get("/")
def root():
    return {
        "message": "Loan Eligibility Prediction API",
        "status": "running"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(application: LoanApplication):
    input_data = pd.DataFrame([application.model_dump()])
    prediction = int(model.predict(input_data)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data)[0][1])

    return {
        "loan_eligibility": prediction,
        "prediction": "Eligible" if prediction == 1 else "Not Eligible",
        "eligibility_probability": probability
    }
