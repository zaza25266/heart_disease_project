
import os
import sqlite3
import joblib
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

load_dotenv()


app = FastAPI(
    title="Heart Disease Clinical Diagnostics API",
    description="An API for predicting heart disease risk based on clinical data.", 
    version="1.0.0"
)

MODEL_PATH = "models/best_Tuned_Logistic_Regression.pkl"