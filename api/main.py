
import os
import sqlite3
import joblib
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

# cross_over_resourse sharing to website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("starting fastapi application...")

MODEL_PATH = "models/best_Tuned_Logistic_Regression.pkl"

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