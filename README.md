# Churn Prediction API (FastAPI + PyCaret)

A simple FastAPI service that loads a trained PyCaret churn model and returns:
- churn probability
- class label (1 = churn, 0 = not churn)
- threshold used for classification

## Endpoints
- GET `/health` → service health check
- POST `/predict` → churn prediction

## Example Request
```bash
curl -X POST "https://churn-api-ujk5.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "contract_Two Year": 0,
    "contract_One Year": 0,
    "referred_a_friend": 0,
    "dependents": 0,
    "senior_citizen": 1,
    "number_of_referrals": 0,
    "offer_Offer D": 0,
    "phone_service_x": 1,
    "premium_tech_support": 0,
    "married": 0
  }'
