import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH")
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def downalod_and_store_data():
    # conform that the local directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    # give names to columns as offlically we dont have
    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
    ]
    print("Downloading raw dataset from UCI Repository...")
    # downalod and load data
    df = pd.read_csv(DATA_URL, header=None, names=columns)

    print("connecting to database...")

    # initialize connection to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # create target tables with manul text to preseve raw values
    cursor.execute(""" 
                  CREATE TABLE IF NOT EXISTS  raw_data(
                      age TEXT, sex TEXT, cp TEXT, trestbps TEXT, chol TEXT,
                      fbs TEXT, restecg TEXT, thalach TEXT, exang TEXT,
                      oldpeak TEXT, slope TEXT, ca TEXT, thal TEXT, num TEXT 
                  )
                  """)
    # empty the table prior to inserting new data to avoid appedning duplicate
    cursor.execute("DELETE FROM raw_data")

    print("inserting data into database...")
    # insert data into database
    df.to_sql("raw_data", conn, if_exists="append", index=False)

    # commout the changes and close the connection
    conn.commit()
    conn.close()
    print("data has been successfully stored in database")


if __name__ == "__main__":
    downalod_and_store_data()
