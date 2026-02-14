from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
def structured_preprocess(data):
    transform_data = scaler.fit_transform(data)
    return transform_data


