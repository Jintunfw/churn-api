from fastapi import FastAPI, HTTPException
import pandas as pd
from pycaret.classification import load_model, predict_model

app = FastAPI()  # <-- MUST be before @app.post

model = load_model("churn_model")  # or "churn_model.pkl" depending on your save

EXPECTED_FEATURES = [
    'contract_Two Year',
    'contract_One Year',
    'referred_a_friend',
    'dependents',
    'senior_citizen',
    'number_of_referrals',
    'offer_Offer D',
    'phone_service_x',
    'premium_tech_support',
    'married',
    'married_senior'
]

THRESHOLD = 0.65

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: dict):
    try:
        df = pd.DataFrame([payload])

        df["married"] = df.get("married", 0).astype(int)
        df["senior_citizen"] = df.get("senior_citizen", 0).astype(int)
        df["married_senior"] = df["married"] * df["senior_citizen"]

        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0

        df = df[EXPECTED_FEATURES]

        preds = predict_model(model, data=df)

        prob_col = next((c for c in ["Score_1", "Score", "prediction_score"] if c in preds.columns), None)
        if prob_col is None:
            raise ValueError(f"No probability column found. Got: {preds.columns.tolist()}")

        prob = float(preds.loc[0, prob_col])
        label = int(prob >= THRESHOLD)

        return {"churn_probability": round(prob, 4), "prediction": label, "threshold": THRESHOLD}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

