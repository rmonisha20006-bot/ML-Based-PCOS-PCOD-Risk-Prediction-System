import streamlit as st
import numpy as np
import joblib
import sklearn

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PCOS Risk Predictor",
    page_icon="🩸",
    layout="wide"
)

# ---------------- MODERN AESTHETIC RED THEME ----------------
st.markdown("""
<style>

/* Full Page Soft Gradient */
.stApp {
    background: linear-gradient(135deg, #fff5f5, #ffeaea);
}

/* Remove excessive margins */
.block-container {
    padding: 2rem 4rem;
}

/* Main Heading */
.stMarkdown h1,
h1 {
    color: #8B0000 !important;
    font-weight: 800 !important;
    text-align: left !important;
    margin-bottom: 0.5rem !important;
}

/* Sub Headings */
.stMarkdown h2,
.stMarkdown h3,
h2, h3 {
    color: #990000 !important;
}

/* Description text */
p {
    font-size: 16px;
    color: #660000 !important;
}

/* Global text color */
.stApp, .stMarkdown, .stText, .block-container, body {
    color: #660000 !important;
}

/* Input label text color */
.css-1d391kg, .css-6y5u6r, .css-14xtw13, .css-1mm2mu1, .css-9s5bis {
    color: #660000 !important;
}

/* Input component text color */
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div,
input,
textarea {
    color: #660000 !important;
}

/* ---------------- WHITE INPUT BOXES ---------------- */
div[data-baseweb="input"] > div {
    background-color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #ffcccc !important;
}

div[data-baseweb="input"] input {
    background-color: #ffffff !important;
}

/* Selectbox Styling */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #ffcccc !important;
}

/* Focus Glow Effect */
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    box-shadow: 0 0 5px rgba(204, 0, 0, 0.4);
}

/* ---------------- BUTTON ---------------- */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #990000, #cc0000);
    color: white !important;
    height: 3em;
    width: 100%;
    font-size: 18px;
    margin-top: 20px;
    border-radius: 12px;
    border: none;
    transition: 0.3s ease;
}

/* Button text override */
div.stButton > button {
    color: white !important;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #660000, #990000);
    transform: scale(1.02);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: #990000;
    color: white;
}

/* Progress Bar*/
.stProgress > div > div > div > div {
    background-color: #990000;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("pcos_model.pkl")

# ---------------- HEADER ----------------
st.markdown("<h1>PCOS/PCOD Risk Prediction System</h1>", unsafe_allow_html=True)

st.markdown("""
<left>
Assess the likelihood of <b>Polycystic Ovary Syndrome (PCOS)/ Polycystic Ovarian Disease (PCOD)</b>
using clinical and menstrual health indicators.
</left>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- CLEAN WHITE SIDEBAR WITH SOFT RED CARDS ----------------
st.sidebar.markdown("""
<style>

/* Sidebar Background - WHITE */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    padding: 25px 15px;
}

/* Sidebar Card - Soft Red */
.sidebar-card {
    background-color: #ffe6e6;
    padding: 15px 18px;
    border-radius: 16px;
    margin-bottom: 20px;
    text-align: justify
    box-shadow: 0 6px 15px rgba(153, 0, 0, 0.15);
    font-size: 14px;
    line-height: 1.6;
    color: #000000;   /* BLACK TEXT */
}

/* Card Titles */
.sidebar-card h4 {
    color: #990000 !important;
    margin-bottom: 8px;
    font-size: 16px;
}

/* Remove extra spacing */
section[data-testid="stSidebar"] .block-container {
    padding-top: 0;
}

</style>
""", unsafe_allow_html=True)

# ---------------- CARD 1 ----------------
st.sidebar.markdown("""
<div class="sidebar-card">
<h4>🩺 About This Application</h4>
This AI-powered system estimates the likelihood of 
<b>Polycystic Ovary Syndrome (PCOS)</b> using clinical and menstrual health indicators.

The model is built using a <b>Random Forest</b> algorithm 
to provide a probability-based risk assessment.
</div>
""", unsafe_allow_html=True)

# ---------------- CARD 2 ----------------
st.sidebar.markdown("""
<div class="sidebar-card">
<h4>⚠ Medical Disclaimer</h4>
• This tool does NOT provide a medical diagnosis.<br>
• For educational purposes only.<br>
• Not a substitute for professional advice.<br>
• Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)

# ---------------- CARD 3 ----------------
st.sidebar.markdown("""
<div class="sidebar-card">
<h4>🔒 Data & Ethics</h4>
• No personal data is stored.<br>
• Predictions are based on trained dataset patterns.<br>
• Accuracy depends on input quality.<br>
• May not generalize to all populations.
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
st.subheader("Patient Clinical Details")

col1, col2 = st.columns(2)

with col1:
    weight = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=150.0,
        step=0.1,
        help="Weight of the patient in kilograms."
    )

    length_cycle = st.number_input(
        "Menstrual Cycle Length (days)",
        min_value=20,
        max_value=60,
        help="Total number of days in one menstrual cycle."
    )

    unusual_bleeding = st.selectbox(
        "Unusual Bleeding",
        ["No", "Yes"],
        help="Indicates whether abnormal bleeding occurs."
    )

with col2:
    length_menses = st.number_input(
        "Duration of Menses (days)",
        min_value=1,
        max_value=15,
        help="Number of days menstrual bleeding lasts."
    )

    num_peaks = st.number_input(
        "Number of Hormonal Peaks",
        min_value=0,
        max_value=5,
        help="Number of hormonal fluctuations observed."
    )

    family_history_option = st.selectbox(
        "Family Medical History",
        ["None", "Diabetes", "PCOS/PCOD", "Thyroid"],
        help="Indicates family history of related hormonal or metabolic disorders."
    )

# ---------------- CONVERT VALUES ----------------
unusual_bleeding = 1 if unusual_bleeding == "Yes" else 0

if family_history_option == "None":
    family_history = 0
elif family_history_option == "Diabetes":
    family_history = 1
elif family_history_option == "PCOS/PCOD":
    family_history = 2
else:
    family_history = 3



# ---------------- PREDICTION ----------------
if st.button("Predict PCOS/PCOD Risk"):

    input_data = np.array([[weight,
                            length_cycle,
                            length_menses,
                            num_peaks,
                            unusual_bleeding,
                            family_history]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠ High Risk of PCOS/PCOD ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of PCOS/PCOD ({probability*100:.2f}%)")

    st.progress(int(probability * 100))