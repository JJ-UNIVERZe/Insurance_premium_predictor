# app.py
import os
os.environ["STREAMLIT_WATCH_FILE"] = "false"

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from src.app.premium import compute_premium
from src.app.explain import plot_shap_for_row, plot_shap_summary  # <-- SHAP functions

# Preload dataset for global SHAP summary
try:
    df_global = pd.read_csv("data/travel_augmented.csv")
except:
    df_global = None

st.set_page_config(
    page_title="Travel Insurance Predictor",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">✈ Travel Safety & Insurance Premium Predictor</h1>', unsafe_allow_html=True)

# Load models with error handling
try:
    clf = joblib.load("models/risk_pipeline.joblib")
    reg = joblib.load("models/claim_pipeline.joblib")
    models_loaded = True
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.info("💡 Ensure models are in the models/ directory")
    models_loaded = False

if models_loaded:
    st.sidebar.markdown("### 🎯 Individual Prediction")
    with st.sidebar.form("prediction_form"):
        st.markdown("Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male","Female","Other"])

        st.markdown("Travel Details")
        mode = st.selectbox("Mode of transport", ["Flight","Train","Bus","Car"])
        trip_length = st.number_input("Trip length (days)", 1, 365, 7)
        dest = st.selectbox("Destination risk", ["Low","Medium","High"])
        purpose = st.selectbox("Travel purpose", ["Leisure","Business","Adventure"])

        st.markdown("Insurance Details")
        coverage = st.number_input("Coverage amount (USD)", 1000, 100000, 5000)
        deductible = st.number_input("Deductible (USD)", 0, 10000, 200)

        st.markdown("Risk Factors")
        prev_claims = st.number_input("Previous claims count", 0, 50, 0)
        preexist = st.selectbox("Preexisting condition?", ["No","Yes"])

        submit = st.form_submit_button("🔮 Get Premium Estimate", use_container_width=True)

    if submit:
        try:
            row = pd.DataFrame([{
                "age": age, "gender": gender, "mode_of_transport": mode,
                "trip_length_days": trip_length, "destination_risk": dest,
                "travel_purpose": purpose, "coverage_amount": coverage,
                "deductible": deductible, "previous_claims_count": prev_claims,
                "has_preexisting_conditions": preexist
            }])

            with st.spinner("🔄 Calculating your premium..."):
                risk_prob = clf.predict_proba(row)[:,1][0]
                expected_claim = float(reg.predict(row)[0])
                premium = compute_premium(risk_prob, expected_claim)

            # Metrics display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🎯 Risk Probability", f"{risk_prob:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("💰 Expected Claim", f"${expected_claim:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("💳 Estimated Premium", f"${premium:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Risk assessment block
            if risk_prob < 0.1:
                risk_level = "🟢 Low Risk"; risk_color = "#28a745"
            elif risk_prob < 0.3:
                risk_level = "🟡 Medium Risk"; risk_color = "#ffc107"
            else:
                risk_level = "🔴 High Risk"; risk_color = "#dc3545"
            st.markdown(f"""
            <div style="background-color: {risk_color}20; border-left: 4px solid {risk_color}; padding: 1rem; border-radius: 0.5rem;">
                <h4 style="color: {risk_color}; margin: 0;">{risk_level}</h4>
                <p style="margin: 0.5rem 0 0 0;">Based on your profile, you have a {risk_prob:.1%} chance of making a claim.</p>
            </div>
            """, unsafe_allow_html=True)

            # --- SHAP Explainability ---
            st.write("*Explainability (SHAP)*")
            img_path = plot_shap_for_row("models/risk_pipeline.joblib", row)
            if img_path:
                st.image(img_path, width='stretch', caption="Top factors for this prediction")
            else:
                st.info("SHAP explanation not available. Model still predicted successfully.")

            # Optional global summary
            if df_global is not None and st.button("Show global feature importance"):
                summary_path = plot_shap_summary("models/risk_pipeline.joblib", df=df_global.sample(50))
                if summary_path:
                    st.image(summary_path, width='stretch', caption="Global Feature Importance (sample)")
                else:
                    st.info("SHAP summary not available.")

        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p><strong>Travel Safety & Insurance Premium Predictor</strong></p>
    <p>Powered by Machine Learning • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)