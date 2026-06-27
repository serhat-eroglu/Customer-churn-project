import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    return df


def preprocess(df):
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    X = df.drop(columns=['Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test