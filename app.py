import streamlit as st
import joblib
import re
import numpy as np

MIN_WORDS = 20
THRESHOLD = 0.4

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

SUSPICIOUS_PATTERNS = {
    "Upfront payment or fee asked": r"registration fee|security deposit|processing fee|training fee|refundable|pay (?:a |the )?(?:fee|amount|deposit)|deposit (?:money|amount)|send (?:money|payment)",
    "Bank or personal details asked": r"bank (?:account )?details|account number|\botp\b|aadhaar|aadhar|pan card|credit card|debit card",
    "Unrealistic salary or easy money promise": r"earn (?:up to )?(?:rs\.?|₹|\$)?\s*\d[\d,]*\s*(?:per|/|a)?\s*(?:day|daily|week|weekly|hour)|guaranteed (?:job|income|salary|placement)|100% (?:job|placement)|no experience (?:needed|required)|easy money|weekly payouts?",
    "Urgency pressure": r"urgent(?:ly)?|limited (?:seats|slots)|apply immediately|hurry|today only",
    "Contact only via chat app, short link or form": r"whatsapp|telegram|bit\.ly|tinyurl|t\.me|wa\.me|forms\.gle|docs\.google\.com/forms",
    "Free email address used for applying": r"@(?:gmail|yahoo|outlook|hotmail)\.com",
}

def find_signals(text):
    found = []
    for label, pattern in SUSPICIOUS_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(label)
    return found

def find_missing(job_text, company_text):
    missing = []
    if company_text.strip() == "":
        missing.append("Company details are missing")
    if not re.search(r"responsibilit|duties|you will|role involves|key tasks", job_text, flags=re.IGNORECASE):
        missing.append("Job responsibilities are not clearly mentioned")
    if not re.search(r"salary|\bctc\b|\blpa\b|stipend|per month|per annum|compensation|₹|\brs\b|\$", job_text, flags=re.IGNORECASE):
        missing.append("Salary details are missing")
    return missing

def explain_words(vector, base_prob):
    row = vector.toarray()[0]
    present = row.nonzero()[0]
    if len(present) == 0:
        return []
    candidates = sorted(present, key=lambda i: row[i], reverse=True)[:30]
    matrix = np.tile(row, (len(candidates), 1))
    for r, i in enumerate(candidates):
        matrix[r, i] = 0
    probs = rf_model_smote.predict_proba(matrix)[:, 1]
    results = [(feature_names[i], base_prob - p) for i, p in zip(candidates, probs)]
    results = [x for x in results if abs(x[1]) > 0.0005]
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results[:5]

st.title("Fake Job Posting Detector")
st.write("Paste a job posting below to check if it looks fraudulent.")

company_text = st.text_area("Company Profile (optional)", height=100)
job_text = st.text_area("Job Posting Text", height=200)

if st.button("Check"):
    if job_text.strip() == "":
        st.warning("Please enter the job posting text.")
    elif len(job_text.split()) < MIN_WORDS:
        st.warning(f"Text is too short. Please paste the full job description (at least {MIN_WORDS} words) for a reliable result.")
    else:
        user_input = job_text + " " + company_text
        cleaned = clean_text(user_input)
        vector = tfidf_vectorizer.transform([cleaned])
        probability = rf_model_smote.predict_proba(vector)[:, 1][0]

        if probability >= THRESHOLD:
            st.error("FRAUD DETECTED ⚠️")
        else:
            st.success("LIKELY GENUINE ✅")
        st.progress(float(probability))
        st.write(f"Estimated fraud probability: {probability:.0%}")

        st.subheader("Why this result? (top 5 words)")
        words = explain_words(vector, probability)
        if not words:
            st.info("No strong word-level signals found.")
        for word, delta in words:
            if delta > 0:
                st.write(f"- **{word}**: increases fraud score ({delta*100:+.1f} points)")
            else:
                st.write(f"- **{word}**: decreases fraud score ({delta*100:+.1f} points)")

        st.subheader("Suspicious signals")
        signals = find_signals(user_input)
        if signals:
            for s in signals:
                st.write("- ⚠️ " + s)
        else:
            st.write("No known suspicious patterns found.")

        st.subheader("Missing information")
        missing = find_missing(job_text, company_text)
        if missing:
            for m in missing:
                st.write("- " + m)
        else:
            st.write("Company, responsibilities and salary details are present.")

        st.subheader("Safety recommendation")
        st.info("Verify the employer's official website, company email domain and contact details. Never pay money for a job.")

        st.caption("This prediction is not a guarantee. Verify the employer before sharing money or personal information.")