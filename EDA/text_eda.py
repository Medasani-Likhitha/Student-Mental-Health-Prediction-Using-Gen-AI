from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt

df_text = pd.read_csv("data/unstructured/reddit_mental_health.csv")

df_text["text"].str.len().hist()

wc = WordCloud().generate(" ".join(df_text["text"][:1000]))
plt.imshow(wc)
