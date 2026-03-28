import re
import streamlit as st  
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

st.title("Fake News Detector")
st.write("Enter a News Articles below to check whether it is Fake or Real.")

news_input = st.text_area("News Article:", "")

if st.button("Check News"):
    if news_input.strip():
        cleaned_input = clean_text(news_input)   # ✅ IMPORTANT
        transform_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(transform_input)
        
        if prediction[0] == 1:
            st.success("The News is Real ✅")
        else:
            st.error("The News is Fake ❌")
    else:
        st.warning("Please enter some text to analyze.")
        
    