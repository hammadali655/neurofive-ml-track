import streamlit as st
import pandas as pd
import joblib

# ---------- Page config ----------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Custom CSS: compact, single-screen layout ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #0a1628 0%, #0f2138 100%);
    }

    /* Kill scrolling, pin everything into the viewport */
    html, body {
        overflow: hidden !important;
        height: 100vh !important;
    }
    [data-testid="stAppViewContainer"] {
        height: 100vh;
        overflow: hidden;
    }
    [data-testid="stAppViewContainer"] > .main {
        height: 100vh;
        overflow: hidden;
    }
    [data-testid="stHeader"] {
        height: 0;
        min-height: 0;
    }

    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 700px;
    }

    /* Tighten default gaps between Streamlit elements */
    [data-testid="stVerticalBlock"] {
        gap: 0.35rem !important;
    }
    div[data-testid="column"] {
        gap: 0.2rem !important;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 0.2rem 0 0.6rem 0;
    }
    .hero h1 {
        font-size: 1.65rem;
        font-weight: 800;
        color: #f2f6fc;
        margin-bottom: 0.15rem;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #9fb3c8;
        font-size: 0.82rem;
        max-width: 480px;
        margin: 0 auto;
        line-height: 1.3;
    }

    /* Card container */
    .card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1rem 1.2rem 0.3rem 1.2rem;
        margin-bottom: 0.6rem;
        backdrop-filter: blur(10px);
    }

    .section-label {
        color: #6f8bad;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        margin-bottom: 0.4rem;
    }

    /* Streamlit widget label overrides */
    label, .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #d7e2ee !important;
        font-weight: 500 !important;
        font-size: 0.78rem !important;
        margin-bottom: 0.1rem !important;
    }

    /* Reduce vertical padding on widget wrappers */
    .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    div[data-baseweb="select"] {
        min-height: 2.1rem !important;
    }
    div[data-baseweb="select"] > div {
        min-height: 2.1rem !important;
        background-color: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        color: #f2f6fc !important;
    }
    .stNumberInput input {
        background-color: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        color: #f2f6fc !important;
        padding: 0.3rem 0.6rem !important;
        height: 2.1rem !important;
    }
    .stNumberInput button {
        height: 1.05rem !important;
    }

    .stSlider {
        padding-top: 0.1rem !important;
    }
    .stSlider [data-baseweb="slider"] {
        margin-top: 0.2rem;
    }

    /* Predict button */
    .stButton {
        margin-top: 0.3rem !important;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 700;
        font-size: 0.92rem;
        padding: 0.5rem 0;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
        color: white;
    }

    /* Result cards */
    .result-survive {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
        border: 1px solid rgba(16,185,129,0.4);
        border-radius: 12px;
        padding: 0.7rem 1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    .result-not-survive {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 12px;
        padding: 0.7rem 1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    .result-title {
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .result-survive .result-title { color: #34d399; }
    .result-not-survive .result-title { color: #f87171; }
    .result-sub {
        color: #b8c6d6;
        font-size: 0.8rem;
    }
    .prob-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
        height: 7px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 6px;
    }

    footer, #MainMenu {visibility: hidden;}
    .footnote {
        text-align: center;
        color: #52667e;
        font-size: 0.68rem;
        margin-top: 0.6rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
@st.cache_resource
def load_model():
    return joblib.load('titanic_pipeline.pkl')

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ---------- Hero ----------
st.markdown("""
    <div class="hero">
        <h1>🚢 Titanic Survival Predictor</h1>
        <p>Enter a passenger's details and see what a machine learning model
        trained on real 1912 passenger records predicts about their odds.</p>
    </div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model file 'titanic_pipeline.pkl' not found. Place it in the same folder as app.py.")
    st.stop()

# ---------- Input card ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Passenger Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2,
                           format_func=lambda x: {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}[x])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", min_value=0, max_value=80, value=30)
    fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

with col2:
    sibsp = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=8, value=0)
    parch = st.number_input("Parents / Children Aboard", min_value=0, max_value=6, value=0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"],
                             format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Engineered features (must match training pipeline) ----------
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

input_df = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked],
    'FamilySize': [family_size],
    'IsAlone': [is_alone]
})

# ---------- Predict ----------
predict_clicked = st.button("Predict Survival")

if predict_clicked:
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    survive_pct = probability[1] * 100

    if prediction == 1:
        st.markdown(f"""
            <div class="result-survive">
                <div class="result-title">✅ Likely to Survive</div>
                <div class="result-sub">Estimated survival probability: <strong>{survive_pct:.1f}%</strong></div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{survive_pct}%; background:#34d399;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-not-survive">
                <div class="result-title">❌ Unlikely to Survive</div>
                <div class="result-sub">Estimated survival probability: <strong>{survive_pct:.1f}%</strong></div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{survive_pct}%; background:#f87171;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

st.markdown(
    '<div class="footnote">Logistic Regression pipeline trained on the Titanic dataset · '
    'NeuroFive Solutions ML Track — Week 5</div>',
    unsafe_allow_html=True
)
