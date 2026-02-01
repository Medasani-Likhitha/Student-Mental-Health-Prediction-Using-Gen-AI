from sklearn.preprocessing import LabelEncoder, StandardScaler

df.fillna(method="ffill", inplace=True)

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

scaler = StandardScaler()
df["CGPA"] = scaler.fit_transform(df[["CGPA"]])

y = df["Depression"]
X = df.drop("Depression", axis=1)