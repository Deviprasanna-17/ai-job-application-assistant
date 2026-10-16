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

feature_names = tfidf_vectorizer.get_feature_names_out()
importances = rf_model_smote.feature_importances_

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

        # Feature 1: color result + progress bar
        if probability >= 0.4:
            st.error("FRAUD ⚠️")
        else:
            st.success("NOT FRAUD ✅")
        st.progress(float(probability))
        st.write(f"Fraud probability: {probability:.2%}")

        # Feature 2: top words that influenced the result
        row = vector.toarray()[0]
        present = row.nonzero()[0]
        if len(present) == 0:
            st.info("No known words from the model's vocabulary were found in this text.")
        else:
            top = sorted(present, key=lambda i: importances[i] * row[i], reverse=True)[:5]
            st.subheader("Top words that influenced this result")
            for i in top:
                st.write("- " + feature_names[i])