from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

scaler = StandardScaler()

def structured_preprocess(data):

    data = data.copy()

    # 1️⃣ Drop Timestamp if exists
    if 'Timestamp' in data.columns:
        data = data.drop(columns=['Timestamp'])

    # 2️⃣ Encode categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns

    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    print(data.head())

    # 3️⃣ Scale numeric values
    transformed_data = scaler.fit_transform(data)

    return transformed_data
