import pandas as pd
from EDA.structured_data_eda import structured_eda
from sklearn.model_selection import train_test_split
from src.preprocessing.structured_preprocess import structured_preprocess


# Load dataset
structured_df = pd.read_csv("/Users/apple/PycharmProjects/Student-Mental-Health-Prediction-Using-Gen-AI/Datasets/Student Mental health.csv")


if __name__ == '__main__':
    structured_df.info()
    structured_eda(structured_df)
    X = structured_df.drop(columns = ['Do you have Depression?'])
    y = structured_df['Do you have Depression?'].map({'Yes': 1, "No": 0})
    structured_X_train, structured_X_test, structured_y_train, structured_y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    structured_X_train_scaler = structured_preprocess(structured_X_train)
    structured_X_test_scaler = structured_preprocess(structured_X_test)
    print("X_train : ", structured_X_train_scaler.shape)
    print("X_test : ", structured_X_test_scaler.shape)
    print("y_train : ", structured_y_train.shape)
    print("y_test : ", structured_y_test.shape)

