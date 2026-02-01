from transformers import pipeline

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

emotion_model("I feel very stressed about my exams")


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

summarizer(df_text["text"][0], max_length=100, min_length=30)

generator = pipeline("text-generation", model="gpt2")

generator(
    "A student feeling anxious about exams should be advised to",
    max_length=80
)