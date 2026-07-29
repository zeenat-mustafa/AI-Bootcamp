import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import uuid
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pickle
import uvicorn

# ------------------------------------------------------------------
# SETUP: Train (or load) the model
# ------------------------------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

iris_model = model
feature_names = iris.feature_names


# Input schema
class IrisFeatures(BaseModel):
    features: List[float]

    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }


# ------------------------------------------------------------------
# PART 1: Rate Limiter
# ------------------------------------------------------------------
class RateLimiter:
    def __init__(self, requests_limit: int = 5, window_seconds: int = 60):
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self.requests = {}  # client_id -> list of request timestamps

    def is_rate_limited(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.requests:
            return False
        recent_requests = [t for t in self.requests[client_id] if now - t < self.window_seconds]
        self.requests[client_id] = recent_requests
        return len(recent_requests) >= self.requests_limit

    def add_request(self, client_id: str) -> None:
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []
        self.requests[client_id].append(now)


app = FastAPI(title="Iris Model API with Rate Limiting")
rate_limiter = RateLimiter(requests_limit=5, window_seconds=60)


async def check_rate_limit(request: Request):
    client_id = request.client.host
    if rate_limiter.is_rate_limited(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    rate_limiter.add_request(client_id)


# ------------------------------------------------------------------
# PART 2: Performance Tracking
# ------------------------------------------------------------------
performance_metrics = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "avg_response_time": 0.0,
    "last_updated": None
}


async def update_metrics(features, prediction, response_time):
    global performance_metrics

    performance_metrics["total_requests"] += 1

    if prediction is not None:
        performance_metrics["successful_predictions"] += 1
    else:
        performance_metrics["failed_predictions"] += 1

    n = performance_metrics["total_requests"]
    current_avg = performance_metrics["avg_response_time"]
    performance_metrics["avg_response_time"] = (current_avg * (n - 1) + response_time) / n

    performance_metrics["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------
# PART 3: Prediction Endpoint
# ------------------------------------------------------------------
prediction_logs = []


@app.post("/predict")
async def predict(iris_data: IrisFeatures, request: Request, rate_limit: None = Depends(check_rate_limit)):
    start_time = time.time()
    prediction = None

    try:
        features = np.array(iris_data.features).reshape(1, -1)
        pred = iris_model.predict(features).tolist()
        pred_proba = iris_model.predict_proba(features).tolist()
        prediction = pred

        response = {
            "prediction": pred,
            "probability": pred_proba
        }
    except Exception as e:
        response = {"error": str(e)}

    response_time = time.time() - start_time

    await update_metrics(iris_data.features, prediction, response_time)

    prediction_logs.append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": iris_data.features,
        "prediction": prediction,
        "response_time": response_time
    })
    if len(prediction_logs) > 100:
        prediction_logs.pop(0)

    return response


# ------------------------------------------------------------------
# PART 4: Dashboard
# ------------------------------------------------------------------
@app.get("/dashboard")
async def dashboard():
    return {
        "performance_metrics": performance_metrics,
        "recent_predictions": prediction_logs[-10:],
        "rate_limit_config": {
            "requests_limit": rate_limiter.requests_limit,
            "window_seconds": rate_limiter.window_seconds
        }
    }


# ------------------------------------------------------------------
# Run the application
# ------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)