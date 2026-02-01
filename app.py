import streamlit as st

st.title("AI-Based Student Mental Health Analyzer")

text = st.text_area("Enter text")
if st.button("Analyze"):
    result = emotion_model(text)
    st.write(result)
