import pandas as pd
import seaborn as sns

df = pd.read_csv("Datasets/Student Mental health.csv")
df.info()
df.describe()

sns.countplot(x="Depression", data=df)
sns.boxplot(x="Depression", y="CGPA", data=df)
