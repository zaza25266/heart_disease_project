
import os
import sqlite3
import joblib
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


# import logger
from logs.logger import get_core_logger

load_dotenv()

# initialize logger for api
logger = get_core_logger("FastAPI_Backend")

app = FastAPI(
    title="Heart Disease Clinical Diagnostics API",
    description="An API for predicting heart disease risk based on clinical data.", 
    version="1.0.0"
)

limiter = Limiter(key_func = get_remote_address)
app.state.Limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# cross_over_resourse sharing to website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("starting fastapi application...")

MODEL_PATH = "models/best_Tuned_Random_Forest.pkl"

@lru_cache(maxsize=1)
def load_model():
    # load the trained model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")

    except Exception as e:
        model = None
        logger.error(f"Error loading model: {str(e)}")

# incoming JSON payload
class PatientData(BaseModel):
    age: float
    trestbps: float
    chol: float
    thalach: float
    oldpeak: float
    cp: str
    restecg: str
    slope: str
    thal: str
    
# store logs to SQLite database
def log_prediction_to_db(input_data : dict, prediction: int, probability: float):
    os.makedirs("data", exist_ok=True)
    db_path = os.getenv("DB_PATH", "data/ml_project.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # create table if not exists
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_logs (
            timestamp TEXT, age REAL, trestbps REAL, chol REAL, 
            thalach REAL, oldpeak REAL, cp TEXT, restecg TEXT, 
            slope TEXT, thal TEXT, prediction INTEGER, probability REAL)
    ''')
    
    # insert log entry
    cursor.execute('''
        INSERT INTO prediction_logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(), input_data['age'], input_data['trestbps'], 
        input_data['chol'], input_data['thalach'], input_data['oldpeak'], 
        input_data['cp'], input_data['restecg'], input_data['slope'], 
        input_data['thal'], prediction, probability
    ))
    
    conn.commit()
    conn.close()
    logger.info(f"prediction stored to the database (prediction: {prediction})")
    
# prediction endpoint
@app.post("/predict")
def predict_heart_disease(patient: PatientData):
    logger.info(f"incoming prediction request recieved via POST / predict (patient: {patient})")
    # call model
    model = load_model()
    if model is None:
        logger.info("model not loaded, cannot generate prediction")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # convert pydantic model to dict and then to DataFrame
    data_dict = patient.model_dump()
    df = pd.DataFrame([data_dict])
    
    # apply custom feature engineering
    df["age_chol_ratio"] = df["age"] / df["chol"]
    df["bp_age_interaction"] = df["trestbps"] * df["age"]
    
    try:
        # generate prediction via sklearn pipeline
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])
        
        # Log telemetry to SQLite
        log_prediction_to_db(data_dict, prediction, probability)
        logger.info(f"prediction successful. returning payload (High Risk: {prediction == 1})")
        # return formatted response
        return{
            "prediction_class": prediction,
            "risk_probability": round(probability, 4),
            "clinical_status": "High Risk" if prediction == 1 else "Low Risk",
            "message": "Prediction generated and telemetry logged successfully."
        }
        
    except Exception as e:
        logger.error(f"pipeline execution crashed during prediction : {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
    
# end point with password to see logs
@app.get("/logs")
@limiter.limit("10/minute")
def get_current_prediction_logs(x_api_key: str = Header(None), limit: int = 50):
    
    api_key = os.getenv("API_KEY")
    
    if not api_key or x_api_key != api_key:
        logger.warning("unauthorized access attempt on / logs endpoint")
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    db_path = os.getenv("DB_PATH", "data/ml_project.db")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT {limit}"
        )
        rows = cursor.fetchall()
        conn.close()
        
        columns = ["timestamp", "age", "trestbps", "chol", "thalach",
                   "oldpeak", "cp", "restecg", "slope", "thal",
                   "prediction", "probability"]
        
        logs = [dict(zip(columns, row)) for row in rows]
                                
        logger.info(f"logs endpoint accessed successfully — returned {len(logs)} records")
        return {"total": len(logs), "logs": logs}  
    
    except Exception as e:
        logger.error(f"failed to read prediction logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Could not read logs: {str(e)}")               