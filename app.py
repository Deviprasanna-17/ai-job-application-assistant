import streamlit as st
import joblib
import re
import numpy as np
from pypdf import PdfReader

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #e0f2fe 0%, #fce7f3 50%, #ede9fe 100%);}
.hero {background: linear-gradient(90deg, #7c3aed, #ec4899, #f97316); padding: 28px; border-radius: 22px; color: white; text-align: center; box-shadow: 0 8px 24px rgba(124,58,237,0.35); margin-bottom: 22px;}
.hero h1 {color: white; font-size: 38px; margin: 0; font-weight: 800;}
.hero p {color: #fff7ed; font-size: 18px; margin: 8px 0 0 0;}
.stTextArea textarea {background: #ffffff; border: 2px solid #c4b5fd; border-radius: 12px; font-size: 16px;}
.stButton > button {background: linear-gradient(90deg, #7c3aed, #ec4899); color: white; border: none; border-radius: 12px; padding: 0.6rem 2.2rem; font-weight: 800; font-size: 18px; box-shadow: 0 4px 14px rgba(236,72,153,0.4);}
.stButton > button:hover {color: white; transform: scale(1.03);}
.sec {font-size: 26px; font-weight: 800; color: #4c1d95; margin: 30px 0 10px 0; border-left: 6px solid #ec4899; padding-left: 12px;}
.sub-red {font-size: 20px; font-weight: 800; color: #b91c1c; margin-bottom: 6px;}
.sub-green {font-size: 20px; font-weight: 800; color: #15803d; margin-bottom: 6px;}
.result {display: flex; gap: 26px; align-items: center; padding: 24px; border-radius: 22px; color: white; box-shadow: 0 8px 24px rgba(0,0,0,0.18);}
.result.fraud {background: linear-gradient(135deg, #ef4444, #f97316);}
.result.genuine {background: linear-gradient(135deg, #10b981, #06b6d4);}
.gauge {width: 120px; height: 120px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0;}
.gauge-in {width: 88px; height: 88px; border-radius: 50%; background: white; display: flex; align-items: center; justify-content: center; font-size: 26px; font-weight: 800; color: #111827;}
.r-title {font-size: 34px; font-weight: 800;}
.r-sub {font-size: 19px; margin-top: 6px;}
.chip {display: inline-block; padding: 8px 15px; border-radius: 999px; margin: 4px 6px 4px 0; font-weight: 700; font-size: 16px;}
.chip-red {background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5;}
.chip-green {background: #dcfce7; color: #166534; border: 1px solid #86efac;}
.chip-orange {background: #ffedd5; color: #9a3412; border: 1px solid #fdba74;}
.grid {display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;}
.info {padding: 14px 16px; border-radius: 14px; font-weight: 700; font-size: 17px;}
.ok {background: #dcfce7; color: #166534; border: 1px solid #86efac;}
.no {background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5;}
.safety {background: linear-gradient(135deg, #fde68a, #fdba74); color: #7c2d12; padding: 18px; border-radius: 16px; font-size: 20px; font-weight: 700;}
.note {font-size: 14px; color: #6b7280; margin-top: 8px;}
</style>
""", unsafe_allow_html=True)

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
        ("Company details", company),
        ("Job responsibilities", responsibilities),
        ("Salary details", salary),
        ("Company address / contact", address),
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

def extract_text(file):
    file.seek(0)
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return file.read().decode("utf-8", errors="ignore")

def section(title):
    st.markdown(f"<div class='sec'>{title}</div>", unsafe_allow_html=True)

def chips_html(items, css):
    return "".join(f"<span class='chip {css}'>{text}</span>" for text in items)

st.markdown("<div class='hero'><h1>🛡️ Fake Job Posting Detector</h1><p>Paste or upload a job posting and check if it looks fraudulent</p></div>", unsafe_allow_html=True)

company_text = st.text_area("Company Profile (optional)", height=100)

uploaded = st.file_uploader("Or upload the job posting (PDF or TXT)", type=["pdf", "txt"])
if uploaded is not None and st.session_state.get("last_file") != uploaded.name:
    extracted = extract_text(uploaded)
    st.session_state.last_file = uploaded.name
    st.session_state.job_text = extracted
    if extracted.strip() == "":
        st.warning("No text could be read from this file. It may be a scanned image PDF. Please paste the text instead.")
    else:
        st.success(f"Read {len(extracted.split())} words from {uploaded.name}")

job_text = st.text_area("Job Posting Text", height=200, key="job_text")

if st.button("Check"):
    if job_text.strip() == "" and uploaded is not None:
        job_text = extract_text(uploaded)
    if job_text.strip() == "":
        st.warning("Please upload a PDF or enter the job posting text.")
    elif len(job_text.split()) < MIN_WORDS:
        st.warning(f"Text is too short. Please paste the full job description (at least {MIN_WORDS} words) for a reliable result.")
    else:
        user_input = job_text + " " + company_text
        cleaned = clean_text(user_input)
        vector = tfidf_vectorizer.transform([cleaned])
        probability = rf_model_smote.predict_proba(vector)[:, 1][0]
        pct = round(probability * 100)

        if probability >= THRESHOLD:
            cls, title, gcolor = "fraud", "FRAUD DETECTED ⚠️", "#ef4444"
        else:
            cls, title, gcolor = "genuine", "LIKELY GENUINE ✅", "#10b981"
        st.markdown(
            f"<div class='result {cls}'><div class='gauge' style='background: conic-gradient({gcolor} {pct}%, #e5e7eb 0);'>"
            f"<div class='gauge-in'>{pct}%</div></div>"
            f"<div><div class='r-title'>{title}</div><div class='r-sub'>Estimated fraud probability: {pct}%</div></div></div>",
            unsafe_allow_html=True)

        section("🔍 Top 5 Words Influencing This Prediction")
        words = explain_words(vector, probability)
        if not words:
            st.info("No strong word-level influence found.")
        else:
            inc = [f"{w} ({d*100:+.1f})" for w, d in words if d > 0]
            dec = [f"{w} ({d*100:+.1f})" for w, d in words if d < 0]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='sub-red'>🔺 Increase fraud score</div>", unsafe_allow_html=True)
                st.markdown(chips_html(inc, "chip-red") if inc else "None in top 5", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='sub-green'>🔻 Decrease fraud score</div>", unsafe_allow_html=True)
                st.markdown(chips_html(dec, "chip-green") if dec else "None in top 5", unsafe_allow_html=True)
        st.markdown("<div class='note'>Each value shows how many points the fraud score changes when that word is removed. Words alone do not decide the result. The full job description context matters.</div>", unsafe_allow_html=True)

        section("🚩 Suspicious signals")
        signals = find_signals(user_input)
        if signals:
            st.markdown(chips_html([f"⚠️ {s}" for s in signals], "chip-orange"), unsafe_allow_html=True)
        else:
            st.markdown(chips_html(["✅ No known suspicious patterns found"], "chip-green"), unsafe_allow_html=True)

        section("📋 Information check")
        cards = "".join(
            f"<div class='info {'ok' if ok else 'no'}'>{'✅' if ok else '❌'} {label}: {'Mentioned' if ok else 'Not found'}</div>"
            for label, ok in check_information(job_text, company_text))
        st.markdown(f"<div class='grid'>{cards}</div>", unsafe_allow_html=True)
        st.markdown("<div class='note'>'Mentioned' only means the text contains it. It does not prove the company is genuine.</div>", unsafe_allow_html=True)

        section("🛡️ Safety recommendation")
        st.markdown("<div class='safety'>Verify the employer's official website, company email domain and contact details. Never pay money for a job.</div>", unsafe_allow_html=True)

        st.markdown("<div class='note'>This prediction is not a guarantee. Verify the employer before sharing money or personal information.</div>", unsafe_allow_html=True)