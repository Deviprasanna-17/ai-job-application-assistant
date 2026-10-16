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

def find_suspicious_links(text):
    urls = re.findall(r'(?:https?://|www\.)\S+', text, flags=re.IGNORECASE)
    risky = ["bit.ly", "tinyurl", "t.me", "wa.me", "chat.whatsapp.com", "forms.gle", "docs.google.com/forms"]
    flags = []
    for u in urls:
        for r in risky:
            if r in u.lower():
                flags.append(u)
    if re.search(r'@(gmail|yahoo|outlook|hotmail)\.com', text, flags=re.IGNORECASE):
        flags.append("Free email address (gmail/yahoo etc.) used for applying")
    return urls, list(dict.fromkeys(flags))

tfidf_vectorizer = joblib.load('tfidf_vectorizer_final.pkl')
rf_model_smote = joblib.load('fraud_detector_model_final.pkl')

FRAUD_SAMPLE = "URGENT! Work from home job. Earn $5000 weekly. No experience needed. Send your bank details immediately to claim this offer!"
REAL_SAMPLE = "We are looking for a Software Engineer with 2 years of experience in Python and SQL. You will design and maintain backend services, work with the product team, and take part in code reviews. Bachelor's degree in Computer Science required. Competitive salary and health benefits."

st.title("Fake Job Posting Detector")
st.write("Paste a job description below to check if it's fraudulent.")

if "job_text" not in st.session_state:
    st.session_state.job_text = ""

col1, col2 = st.columns(2)
if col1.button("Try sample fraud posting"):
    st.session_state.job_text = FRAUD_SAMPLE
if col2.button("Try sample real posting"):
    st.session_state.job_text = REAL_SAMPLE

user_input = st.text_area("Job Posting Text", height=200, key="job_text")

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
        urls, flags = find_suspicious_links(user_input)
        if flags:
            st.warning("Suspicious items found in this posting:")
            for f in flags:
                st.write("- " + f)
        elif urls:
            st.info(f"{len(urls)} link(s) found, but no known risky pattern.")