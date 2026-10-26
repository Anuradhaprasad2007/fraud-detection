from pathlib import Path
import joblib
import pandas as pd
import shap
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.database import engine, get_db, Base
from src.models_db import TransactionRecord

# ---- Config: change these 3 things when the model is retrained, nothing else ----
MODEL_PATH = Path(__file__).parent / "models" / "fraud_xgboost_final.pkl"
FEATURE_COLUMNS = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]
THRESHOLD = 0.90
# -----------------------------------------------------------------------------

model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Fraud Detection API")


class Transaction(BaseModel):
    step: int
    type: str  # one of: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


@app.get("/")
def root():
    return {"status": "Fraud Detection API is running"}


@app.post("/predict")
def predict(txn: Transaction, db: Session = Depends(get_db)):
    row = {col: 0 for col in FEATURE_COLUMNS}
    row["step"] = txn.step
    row["amount"] = txn.amount
    row["oldbalanceOrg"] = txn.oldbalanceOrg
    row["newbalanceOrig"] = txn.newbalanceOrig
    row["oldbalanceDest"] = txn.oldbalanceDest
    row["newbalanceDest"] = txn.newbalanceDest

    type_col = f"type_{txn.type}"
    if type_col not in FEATURE_COLUMNS:
        return {"error": f"invalid type '{txn.type}'. Must be one of CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER"}
    row[type_col] = 1

    X = pd.DataFrame([row])[FEATURE_COLUMNS]

    fraud_probability = float(model.predict_proba(X)[0][1])
    is_fraud = fraud_probability >= THRESHOLD

    # SHAP explanation: which features pushed this prediction toward/away from fraud
    shap_values = explainer.shap_values(X)
    contributions = dict(zip(FEATURE_COLUMNS, shap_values[0].tolist()))
    top_factors = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    explanation = [
        {
            "feature": name,
            "impact": round(value, 4),
            "direction": "increased fraud risk" if value > 0 else "decreased fraud risk",
        }
        for name, value in top_factors
    ]

    record = TransactionRecord(
        step=txn.step,
        type=txn.type,
        amount=txn.amount,
        oldbalanceOrg=txn.oldbalanceOrg,
        newbalanceOrig=txn.newbalanceOrig,
        oldbalanceDest=txn.oldbalanceDest,
        newbalanceDest=txn.newbalanceDest,
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "is_fraud": is_fraud,
        "fraud_probability": round(fraud_probability, 4),
        "threshold_used": THRESHOLD,
        "top_factors": explanation,
    }


@app.get("/transactions")
def get_transactions(limit: int = 50, db: Session = Depends(get_db)):
    records = (
        db.query(TransactionRecord)
        .order_by(desc(TransactionRecord.created_at))
        .limit(limit)
        .all()
    )
    return records