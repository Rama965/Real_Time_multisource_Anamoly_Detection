import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb


st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="🔧",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.pinimg.com/736x/44/23/65/442365f83c6e9b6e3248e2e67ca2a3fe.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .stButton>button {
        background-color: blue;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .stNumberInput>div>div>input {
        background-color: rgba(255,255,255,0.8);
        border-radius: 5px;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("machine_model.json")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("model_features.pkl")
    return model, scaler, features

model, scaler, feature_names = load_model()


#st.title("🔧 Machine Failure Prediction Dashboard")
st.markdown("<h1 style='text-align: center;'>🔧 Machine Failure Prediction Dashboard</h1>", unsafe_allow_html=True)
#st.write("Enter machine sensor readings below to predict **failure risk**.")
st.markdown(
    "<p style='text-align:center; font-size:18px;'>Enter machine sensor readings below to predict <b>failure risk</b>.</p>",
    unsafe_allow_html=True
)



input_values = {}


int_limited_cols = {
    'Maintenance_History_Count': (0, 20),
    'Failure_History_Count': (0, 20),
    'Error_Codes_Last_30_Days': (0, 20),
    'AI_Override_Events': (0, 20)
}
col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    target_col = col1 if i % 2 == 0 else col2

    if feature in int_limited_cols:
        mn, mx = int_limited_cols[feature]
        input_values[feature] = target_col.slider(
            feature, min_value=mn, max_value=mx, value=0
        )
    else:
        input_values[feature] = target_col.number_input(feature, value=0.0, format="%.2f")

input_df = pd.DataFrame([input_values])
scaled_input = scaler.transform(input_df)

st.markdown("---")

if st.button("Predict Machine Failure", use_container_width=True):
    try:
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        colA, colB = st.columns(2)
        colA.metric("Prediction", "FAILURE" if prediction == 1 else "NO FAILURE")
        colB.metric("Failure Probability", f"{probability*100:.2f}%")

        if prediction == 1:
            st.error("Machine is going to fail within 7 days!")
        else:
            st.success("✅ Machine is operating normally.")

    except Exception as e:
        st.error(f"Error: {e}")