# pipleline/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="data/iris.csv"):
    return pd.read_csv(path)

def preprocess_data(df):
    X = df.drop("target",axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

