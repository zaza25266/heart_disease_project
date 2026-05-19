import os
import numpy as np
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# load environment variables
load_dotenv()
DB_PATH = os.getenv("DB_PATH")


def load_and_process_data():
    # Connect to the SQLite database
    conn = sqlite3.connect(DB_PATH)

    # Load data into a DataFrame
    query = "SELECT * FROM raw_data"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # clean missing values
    df.replace("?", np.nan, inplace=True)

    # convert raw text into numerical features
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # feature engenring _ inject medical domain signals
    df["age_chol_ratio"] = df["age"] / (df["chol"] + 1e-5)  # avoid division by zero
    df["bp_age_interaction"] = df["age"] * df["trestbps"]

    # fix target column (convert multi_class to binary)
    df["num"] = pd.to_numeric(df["num"], errors="coerce")
    df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
    df.drop(columns=["num"], inplace=True)

    # seperate inputs and target
    X = df.drop(columns=["target"])
    y = df["target"]

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline():
    # Define strict input lists for structural transformations
    numeric_features = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
        "age_chol_ratio",
        "bp_age_interaction",
    ]
    categorical_features = ["cp", "restecg", "slope", "thal"]

    # build pipeline for different datatype
    numeric_transformer = Pipeline(
        steps=[("imputer", KNNImputer(n_neighbors=5)), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # combine processing paths using column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # drop any columns not specified in the transformers
    )

    # use light random_forest to select the most important features
    feature_selector = SelectFromModel(
        estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        max_features=10,
    )

    # combine preprocessor with feature selection in a single pipeline
    preprocessing_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("feature_selection", feature_selector)]
    )

    print("Data loading and preprocessing completed successfully.")

    return preprocessing_pipeline


if __name__ == "__main__":
    load_and_process_data()
    build_preprocessing_pipeline()
