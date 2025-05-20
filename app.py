from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
from houseProject.pipeline.prediction import PredictionPipeline
from datetime import date
import json
from houseProject.utils.common import load_location_encodings

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define expected input schema for prediction
class PredictionInput(BaseModel):
    house_type: int
    city: str
    location: str
    numBathrooms: float
    numBalconies: float
    isNegotiable: float
    verificationDate: date
    SecurityDeposit: float
    Status: float
    bhk: float
    house_size_sqft: float

@app.get("/")
async def health_check():
    return {"status": "success", "message": "Real Estate Price Prediction API is live."}

@app.get("/train")
async def train_model():
    try:
        os.system("python main.py")  # or call a function directly
        return {"status": "success", "message": "Model training completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Load city mapping
city_mapping = {
    'Delhi': 0,
    'Pune': 1,
    'Mumbai': 2,
}


location_encodings = load_location_encodings()

@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        # Map city to numeric encoding
        city_encoded = city_mapping.get(data.city)
        if city_encoded is None:
            raise HTTPException(status_code=400, detail=f"Unsupported city: {data.city}")

        # Map location to numeric encoding using the appropriate city dictionary
        location_dict = location_encodings.get(data.city)
        location_encoded = location_dict.get(data.location)
        if location_encoded is None:
            raise HTTPException(status_code=400, detail=f"Unsupported location '{data.location}' for city '{data.city}'")

        # Convert date to number of days since verification
        days_since_verification = (date.today() - data.verificationDate).days

        # Optional: Map house_type if your model expects it as encoded (you may need a dictionary for this)

        # Build input array for the model in expected order
        input_data = [
            data.house_type,            
            data.house_size_sqft,
            location_encoded,
            city_encoded,
            data.numBathrooms,
            data.numBalconies,
            data.isNegotiable,
            days_since_verification,
            data.SecurityDeposit,
            data.Status,
            data.bhk
        ]

        input_array = np.array(input_data).reshape(1, -1)

        pipeline = PredictionPipeline()
        predicted_log_price = pipeline.predict(input_array)
        prediction = np.exp(predicted_log_price)

        return {"status": "success", "predicted_price": float(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
