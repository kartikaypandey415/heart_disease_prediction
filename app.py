import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Keep these three model files in the same folder as this app.py.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "knn_heart_model.pkl"
SCALER_PATH = BASE_DIR / "heart_scaler.pkl"
COLUMNS_PATH = BASE_DIR / "heart_columns.pkl"

st.set_page_config(
    page_title="Heart Health Check",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f7fbff 0%, #f8f7ff 55%, #fff8f7 100%); }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero { padding: 2rem 2.2rem; border-radius: 24px; color: white;
            background: linear-gradient(115deg, #102a43 0%, #174a68 58%, #6b4d86 100%);
            box-shadow: 0 18px 50px rgba(20, 54, 82, .16); margin-bottom: 1.5rem; }
    .eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: .75rem; font-weight: 700; color: #bde8f7; }
    .hero h1 { font-family: 'Manrope', sans-serif; font-size: clamp(2rem, 4vw, 3.1rem); line-height: 1.1; margin: .55rem 0 .7rem; color: white; }
    .hero p { font-size: 1.04rem; max-width: 700px; color: #e2eef7; margin-bottom: 0; }
    .section-title { font-family: 'Manrope', sans-serif; color: #142f46; font-size: 1.3rem; font-weight: 800; margin: .4rem 0 .2rem; }
    .muted { color: #64748b; font-size: .93rem; }
    div[data-testid="stForm"] { background: rgba(255,255,255,.88); border: 1px solid #e5eaf2; border-radius: 20px; padding: 1.35rem 1.4rem; }
    div[data-testid="stFormSubmitButton"] button { background: linear-gradient(100deg, #0f766e, #167d9a); color: white; border: 0; border-radius: 12px; min-height: 3rem; font-weight: 700; width: 100%; }
    div[data-testid="stFormSubmitButton"] button:hover { background: #115e59; color: white; }
    .result-card { background: white; border: 1px solid #e6eaf0; border-radius: 20px; padding: 1.5rem; box-shadow: 0 12px 35px rgba(15, 35, 55, .06); }
    .footnote { color: #64748b; font-size: .82rem; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e6eaf0; border-radius: 15px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

def load_artifacts():
    missing = [p.name for p in (MODEL_PATH, SCALER_PATH, COLUMNS_PATH) if not p.exists()]
    if missing:
        st.error("Model files missing: " + ", ".join(missing) + ". Put all three .pkl files beside app.py.")
        st.stop()
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(COLUMNS_PATH, "rb") as f:
            columns = pickle.load(f)
        return model, scaler, columns
    except Exception as exc:
        st.error(f"Could not load the saved model files: {exc}")
        st.stop()

model, scaler, feature_columns = load_artifacts()

st.markdown("""
<div class="hero">
  <div class="eyebrow">Heart health • ML-assisted screening</div>
  <h1>A clearer first look<br>at your heart health.</h1>
  <p>Enter the health measurements below to see the prediction from your saved machine-learning model.
  This is an educational screening aid—not a diagnosis.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.65, 1], gap="large")
with left:
    st.markdown('<div class="section-title">Your health details</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Use values from a recent medical check-up when possible.</div>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.markdown("**01 · Basic information**")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", min_value=18, max_value=110, value=45, step=1)
        with c2:
            sex = st.selectbox("Sex", ["Female", "Male"])
        with c3:
            fasting_bs = st.selectbox("Fasting blood sugar", ["≤ 120 mg/dL", "> 120 mg/dL"])

        st.markdown("**02 · Heart and exercise measurements**")
        c1, c2, c3 = st.columns(3)
        with c1:
            resting_bp = st.number_input("Resting blood pressure (mm Hg)", min_value=60, max_value=250, value=120, step=1)
        with c2:
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=700, value=200, step=5,
                                          help="Enter the measured value. A value of 0 may mean the result was not recorded in the source dataset.")
        with c3:
            max_hr = st.number_input("Maximum heart rate", min_value=40, max_value=230, value=150, step=1)
        c4, c5 = st.columns(2)
        with c4:
            oldpeak = st.number_input("ST depression (Oldpeak)", min_value=-3.0, max_value=10.0, value=0.0, step=0.1, format="%.1f")
        with c5:
            exercise_angina = st.selectbox("Exercise-induced angina", ["No", "Yes"])

        st.markdown("**03 · Clinical test categories**")
        c1, c2, c3 = st.columns(3)
        with c1:
            chest_pain = st.selectbox("Chest pain type", [
                "ASY — Asymptomatic", "ATA — Atypical angina",
                "NAP — Non-anginal pain", "TA — Typical angina"
            ])
        with c2:
            resting_ecg = st.selectbox("Resting ECG", [
                "LVH — Left ventricular hypertrophy", "Normal", "ST — ST-T wave abnormality"
            ])
        with c3:
            st_slope = st.selectbox("ST slope", ["Down", "Flat", "Up"])

        submitted = st.form_submit_button("Calculate model prediction  →", use_container_width=True)

with right:
    st.markdown('<div class="section-title">How to use this</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="result-card">
      <p style="font-size:1.05rem;font-weight:700;color:#16324a;margin-top:0">A simple 3-step check</p>
      <p>① Enter values from your report.</p>
      <p>② Select the categories that match your report.</p>
      <p>③ Run the model to view its output.</p>
      <hr style="border:0;border-top:1px solid #e8edf3;margin:1rem 0">
      <p style="font-weight:700;color:#16324a">What this tool does</p>
      <p class="muted">It prepares the input in the feature order expected by your saved model, scales the values, and runs the KNN classifier.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.info("If you have chest pain, shortness of breath, fainting, or other urgent symptoms, seek medical care immediately. Do not use this app to decide whether care is needed.")

if submitted:
    # One-hot encoding mirrors the feature columns saved with the model.
    row = {col: 0 for col in feature_columns}
    values = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs.startswith(">") else 0,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "Male" else 0,
        "ChestPainType_ATA": int(chest_pain.startswith("ATA")),
        "ChestPainType_NAP": int(chest_pain.startswith("NAP")),
        "ChestPainType_TA": int(chest_pain.startswith("TA")),
        "RestingECG_Normal": int(resting_ecg == "Normal"),
        "RestingECG_ST": int(resting_ecg.startswith("ST")),
        "ExerciseAngina_Y": int(exercise_angina == "Yes"),
        "ST_Slope_Flat": int(st_slope == "Flat"),
        "ST_Slope_Up": int(st_slope == "Up"),
    }
    unknown = [key for key in values if key not in row]
    if unknown:
        st.error("The app's feature mapping does not match the model's saved columns: " + ", ".join(unknown))
    else:
        row.update(values)
        input_df = pd.DataFrame([[row[c] for c in feature_columns]], columns=feature_columns)
        try:
            scaled = scaler.transform(input_df)
            prediction = model.predict(scaled)[0]
            probability = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(scaled)[0]
                classes = list(model.classes_)
                positive_candidates = [i for i, label in enumerate(classes) if str(label).lower() in ("1", "true", "yes", "heart disease", "disease")]
                positive_idx = positive_candidates[0] if positive_candidates else (1 if len(classes) > 1 else 0)
                probability = float(probs[positive_idx])

            st.markdown("---")
            st.markdown('<div class="section-title">Your model result</div>', unsafe_allow_html=True)
            has_disease = str(prediction).lower() in ("1", "true", "yes", "heart disease", "disease")
            if has_disease:
                st.error("Model output: Higher likelihood of heart disease")
                st.write("The model classified this input in its heart-disease-positive class. This is not a diagnosis; please discuss your measurements with a qualified healthcare professional.")
            else:
                st.success("Model output: Lower likelihood of heart disease")
                st.write("The model classified this input in its heart-disease-negative class. A negative model result does not rule out heart disease or replace medical advice.")
            if probability is not None:
                st.metric("Model-estimated probability for positive class", f"{probability:.1%}")
                st.caption("This is the model's estimated class probability, not a calibrated personal risk estimate.")
            with st.expander("See the values sent to the model"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Prediction failed. Check that the model, scaler, and feature columns were saved from the same training pipeline. Details: {exc}")

st.markdown("---")
st.markdown("""
<div class="footnote">
<strong>Important:</strong> This tool is for educational demonstration only. It is not medical advice,
a diagnostic device, or a substitute for a clinician's assessment. Predictions can be wrong and depend
on the quality and range of the training data. Do not delay urgent care based on this result.
</div>
""", unsafe_allow_html=True)
