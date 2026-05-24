import os
import sqlite3
import pandas as pd 
from dotenv import load_dotenv
from pathlib import Path 

# Load environment configurations
load_dotenv()

# Grab the database path from .env
DB_PATH = Path(os.getenv('DB_PATH', 'data/ml_project.db')).resolve()

# The exact URLs for all 4 hospital datasets
urls = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
]


def download_and_store_data():
    # confirm that the target local directory folder physically exists
    if os.path.dirname(DB_PATH):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
    print(f"Database Target Configuration Path: {DB_PATH}")
    print(f"Absolute Path Destination: {os.path.abspath(DB_PATH)}")

    # Official clinical attribute headers matching the UCI repository layout
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    
    dataframes = []
    
    print("\nStep 1: Downloading raw dataset from UCI Repository...")
    try:
        for url in urls:
            single_df = pd.read_csv(url, header=None, names=columns, na_values="?")
            dataframes.append(single_df)
        # total dataset
        df = pd.concat(dataframes, ignore_index=True)
        print(f"Successfully downloaded {df.shape[0]} patient rows from UCI.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to download data from UCI web server! Details: {e}")
        return
    
    print("\nConnecting to SQLite storage lake...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
   
    # Keeping everything as TEXT to preserve original strings and missing '?' marks safely
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS raw_data (
            age TEXT, sex TEXT, cp TEXT, trestbps TEXT, chol TEXT,
            fbs TEXT, restecg TEXT, thalach TEXT, exang TEXT,
            oldpeak TEXT, slope TEXT, ca TEXT, thal TEXT, num TEXT 
        )
    ''')
    
    # Empty the table prior to inserting new data to avoid appending duplicate datasets on re-runs
    cursor.execute("DELETE FROM raw_data")
    conn.commit()
    
    print("Pushing raw matrix strings into 'raw_data' table...")
    # Insert data frame into SQL lake
    df.to_sql("raw_data", conn, if_exists="replace", index=False)
    
    # Commit changes and terminate connection safely
    conn.commit()
    conn.close()
    print("SUCCESS: Raw data layer successfully built and locked down!")

if __name__ == "__main__":
    download_and_store_data()