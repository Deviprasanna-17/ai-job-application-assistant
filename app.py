import streamlit as st
import joblib
import re
import numpy as np
from pypdf import PdfReader

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

def has(pattern, text):
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

def check_information(job_text, company_text):
    combined = job_text + " " + company_text
    company = company_text.strip() != "" or has(
        r"about (?:us|the company)|company (?:profile|overview)|who we are|\b(?:pvt|private|ltd|limited|llp|inc|corp|technologies|solutions|systems|services)\b",
        combined)
    responsibilities = has(r"responsibilit|duties|you will|role involves|key tasks", combined)
    salary = has(r"salary|\bctc\b|\blpa\b|stipend|per month|per annum|compensation|₹|\brs\b|\$", combined)
    address = has(r"address|located (?:at|in)|office (?:at|in)|\b\d{6}\b|\+?\d[\d\s\-]{9,}|@[a-z0-9\-]+\.[a-z]{2,}", combined)
    return [
        {"Information": "Company details", "Status": "✅ Mentioned" if company else "❌ Not found"},
        {"Information": "Job responsibilities", "Status": "✅ Mentioned" if responsibilities else "❌ Not found"},
        {"Information": "Salary details", "Status": "✅ Mentioned" if salary else "❌ Not found"},
        {"Information": "Company address / contact", "Status": "✅ Mentioned" if address else "❌ Not found"},
    ]

def explain_words(vector, base_prob, cap=250):
    row = vector.toarray()[0]
    present = row.nonzero()[0]
    if len(present) == 0:
        return []
    if len(present) > cap:
        present = present[np.argsort(row[present])[::-1][:cap]]
    matrix = np.tile(row, (len(present), 1))
    matrix[np.arange(len(present)), present] = 0
    probs = rf_model_smote.predict_proba(matrix)[:, 1]
    deltas = base_prob - probs
    results = [(feature_names[i], d) for i, d in zip(present, deltas) if abs(d) > 0.0005]
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results[:5]

def big_box(text, bg, size):
    st.markdown(
        f"<div style='background:{bg};padding:18px;border-radius:10px;"
        f"font-size:{size}px;font-weight:700;color:white;text-align:center'>{text}</div>",
        unsafe_allow_html=True)
def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return file.read().decode("utf-8", errors="ignore")

st.title("Fake Job Posting Detector")
st.write("Paste a job posting below to check if it looks fraudulent.")

company_text = st.text_area("Company Profile (optional)", height=100)
uploaded = st.file_uploader("Or upload the job posting (PDF or TXT)", type=["pdf", "txt"])
if uploaded is not None and st.session_state.get("last_file") != uploaded.name:
    extracted = extract_text(uploaded)
    st.session_state.last_file = uploaded.name
    st.session_state.job_text = extracted
    if extracted.strip() == "":
        st.warning("No text could be read from this file. It may be a scanned image PDF. Please paste the text instead.")

job_text = st.text_area("Job Posting Text", height=200, key="job_text")

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
            big_box("FRAUD DETECTED ⚠️", "#c62828", 34)
        else:
            big_box("LIKELY GENUINE ✅", "#2e7d32", 34)
        st.progress(float(probability))
        st.markdown(f"## Estimated fraud probability: {probability:.0%}")

        st.markdown("## Top 5 Words Influencing This Prediction")
        words = explain_words(vector, probability)
        if not words:
            st.info("No strong word-level influence found.")
        else:
            inc = [(w, d) for w, d in words if d > 0]
            dec = [(w, d) for w, d in words if d < 0]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 🔺 Increase fraud score")
                if inc:
                    for w, d in inc:
                        st.markdown(f"**{w}**  ({d*100:+.1f} points)")
                else:
                    st.write("None in top 5")
            with c2:
                st.markdown("### 🔻 Decrease fraud score")
                if dec:
                    for w, d in dec:
                        st.markdown(f"**{w}**  ({d*100:+.1f} points)")
                else:
                    st.write("None in top 5")
        st.caption("Each value shows how much the fraud score changes when that word is removed. "
                   "Words alone do not decide the result. The full job description context matters.")

        st.markdown("## Suspicious signals")
        signals = find_signals(user_input)
        if signals:
            for s in signals:
                st.markdown(f"### ⚠️ {s}")
        else:
            st.write("No known suspicious patterns found.")

        st.markdown("## Information check")
        st.table(check_information(job_text, company_text))
        st.caption("'Mentioned' only means the text contains it. It does not prove the company is genuine.")

        st.markdown("## Safety recommendation")
        big_box("Verify the employer's official website, company email domain and contact details. Never pay money for a job.", "#ef6c00", 20)

        st.caption("This prediction is not a guarantee. Verify the employer before sharing money or personal information.")