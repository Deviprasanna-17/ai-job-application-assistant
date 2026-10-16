import streamlit as st
import joblib
import re

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = re.sub('&amp;', 'and', text)
    text = re.sub('[^a-zA-Z\s]', '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text

tfidf_vectorizer = joblib.load('tfidf_vectorizer_final.pkl')
rf_model_smote = joblib.load('fraud_detector_model_final.pkl')

st.title("Fake Job Posting Detector")
st.write("Paste a job description below to check if it's fraudulent.")

user_input = st.text_area("Job Posting Text", height=200)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf_vectorizer.transform([cleaned])
        probability = rf_model_smote.predict_proba(vector)[:, 1][0]
        prediction = "FRAUD ⚠️" if probability >= 0.4 else "NOT FRAUD ✅"
        st.subheader(prediction)
        st.write(f"Fraud probability: {probability:.2%}")