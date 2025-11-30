import streamlit as st
import numpy as np
import pandas as pd
import joblib 

model = joblib.load("machine_model.pkl")
scaler = joblib.load("scaler.pkl")

columns=['Temperature_C', 'Vibration_mms', 'Sound_dB', 'Oil_Level_pct',
       'Coolant_Level_pct', 'Power_Consumption_kW',
       'Last_Maintenance_Days_Ago', 'Maintenance_History_Count',
       'Failure_History_Count', 'Error_Codes_Last_30_Days',
       'Remaining_Useful_Life_days', 'AI_Override_Events']

def user_input_features():
   features = {}
   for col in columns:
       if col in ["Temperature_C" ,"Vibration_mms" ,"Sound_dB" ,"Oil_Level_pct", "Coolant_Level_pct", "Power_Consumption_kW", "Last_Maintenance_Days_Ago", "Remaining_Useful_Life_days"]: 
          st.number_input(col,min_value=0.0,format="%.2f")
       elif col in ["Maintenance_History_Count","Failure_History_Count","Error_Codes_Last_30_Days"]:
           features[col] = st.multiselect(col,[1,2,3,4,5,6])
       elif col in ["Error_Codes_Last_30_Days"]:
           features[col] = st.multiselect(col,[0,1,2,3,4,5,6])
   return features


def preprocess_input(input_dict):
    # build array from the dict in the same order as 'columns'
    arr = [input_dict[col] for col in columns]

    # convert to numpy array
    arr_np = np.array(arr, dtype=float).reshape(1, -1)

    # indices of numeric columns in 'columns'
    num_cols = [
        "Temperature_C", "Vibration_mms", "Sound_dB",
        "Oil_level_pct", "Coolant_level_pct", "Power_Consumption_kw",
        "Last_Maintenance_Days_Ago", "Remaining_Useful_Life_days"
    ]
    num_index = [columns.index(x) for x in num_cols]

    # scale only numeric columns
    arr_np[:, num_index] = scaler.transform(arr_np[:, num_index])

    return arr_np


st.title("Real Time MultiSource Anamoly Detection and Decision Making Platform For Industrial IOT")
st.write("Enter Machine Details To Check")

user_input = user_input_features()

if st.button("Enter"):
    input_prepared = preprocess_input(user_input)
    pred = model.predict(input_prepared)
    result = "Failure_Within_7_Days" if pred[0] == 1 else "Machine is perfectly Alright"
    st.success(f"Prediction: {result}")