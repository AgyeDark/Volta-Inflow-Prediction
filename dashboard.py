import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Load Your Saved Model and Scalers ---
# These files should be in the same folder as this script.
model = load_model('final_lstm_model.keras')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

st.title('🌊 Lake Volta Inflow Prediction Dashboard')

st.write("This model uses the last 30 days of data to predict the river discharge 7 days into the future.")

# --- User Input (Simulation) ---
st.subheader("Simulate a Prediction Using Recent Data")
# In a real-world app, this would fetch live data.
# For our project, we use the last 30 days of our test set as a stand-in.
X_test = pd.read_csv('data/processed/X_test.csv', index_col='date', parse_dates=True)
last_30_days = X_test.tail(30)

st.write("Using the last 30 days of available data as input:")
st.dataframe(last_30_days)

if st.button('🔮 Predict 7-Day Ahead Inflow'):
    # --- The Prediction Engine ---
    # 1. Scale the input data using the loaded scaler
    input_scaled = scaler_X.transform(last_30_days)
    
    # 2. Reshape the data into a single sequence for the LSTM
    input_sequence = np.array([input_scaled])
    
    # 3. Make the prediction
    prediction_scaled = model.predict(input_sequence)
    
    # 4. Inverse transform the prediction to get the real value
    prediction_real_units = scaler_y.inverse_transform(prediction_scaled)
    
    # --- Display the Result ---
    st.success(f"Predicted Inflow in 7 Days: **{prediction_real_units[0][0]:.2f} m³/s**")
    