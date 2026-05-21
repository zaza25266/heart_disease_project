# Heart Disease Risk Predictor
**Author:** Zubair Ali

🌐 **Live Demo:** https://heart-disease-project-14d0.onrender.com/

> First request may take 30–60 seconds — hosted on Render free tier.

⚠️ **Note:** This is a learning project. Built to understand the full MLOps workflow from training to deployment. Not production-grade.

---

## What is this?

A machine learning pipeline that takes a patient's clinical data (age, cholesterol, blood pressure, etc.) and predicts whether they are at risk of heart disease.

The goal wasn't just to train a model. The goal was to build the full pipeline — data, training, testing, API, Docker, CI, and cloud deployment — the way it's actually done in real ML engineering.

---

## How it was built

**1. Data Exploration**
Loaded the UCI Heart Disease dataset, looked at feature distributions, found missing values, and understood what the data actually means before touching any model.

**2. Preprocessing**
Cleaned the data, scaled numeric columns, and created two new features (`age_chol_ratio` and `bp_age_interaction`) to give the model better signals.

**3. Training & Tuning**
Trained both Logistic Regression and Random Forest. Used GridSearchCV with Cross-Validation to find the best settings for each. Picked Logistic Regression in the end — it matched Random Forest on recall but runs faster.

**4. Experiment Tracking**
Every training run, metric, and parameter was logged with MLflow. This makes it easy to compare runs and know exactly why the final model was chosen.

**5. Testing**
Wrote Pytest tests to check data integrity and model output. GitHub Actions runs these tests automatically on every push.

**6. API & Deployment**
Built a FastAPI backend that loads the trained model and returns predictions. Wrapped it in Docker and deployed to Render.

---

## Tech Stack

| Area | Tools |
|---|---|
| ML & Data | Python, Scikit-Learn, Pandas, MLflow, Joblib |
| Backend | FastAPI, Uvicorn, SQLite |
| MLOps | Docker, Docker Compose, GitHub Actions, Pytest |
| Frontend | HTML, CSS, Vanilla JS |

---

## Project Structure

```text
heart_disease_project/
├── .github/
│   └── workflows/
│       └── ci.yml                         # Runs tests on every push
├── api/
│   └── main.py                            # FastAPI backend
├── data/                                  # SQLite prediction logs
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── logs/
│   └── logger.py
├── mlartifacts/                           # MLflow run history
├── models/
│   └── best_Tuned_Logistic_Regression.pkl # Final model
├── notebook/
│   └── eda.ipynb                          # Data exploration
├── preprocess_pipeline/
│   ├── __init__.py
│   └── pre_processing.py                  # Feature engineering
├── py_test/
│   ├── test_data.py                       # Data integrity tests
│   └── test_predict.py                    # Prediction output tests
├── training/
│   └── train.py                           # Training & tuning script
├── Dockerfile.api
├── docker-compose.yml
├── requirements.txt
└── setup.py
```