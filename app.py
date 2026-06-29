import streamlit as st
import pandas as pd
import joblib
from datetime import date

st.set_page_config(page_title="Flight Price Prediction", page_icon="✈️", layout="centered")

st.markdown("""
    <div style='text-align: center; margin-bottom: 20px; padding-top: 20px;'>
        <h1 style='color: #1E3A8A; font-family: sans-serif;'>Dynamic Flight Price Radar</h1>
        <div style='display: flex; align-items: center; justify-content: center; font-size: 22px; color: #4B5563; margin-top: 15px;'>
            <span style='margin: 0 20px; color: #3B82F6; letter-spacing: 3px;'>- - - - - - - - - ✈︎ - - - - - - - - -</span>
        </div>
        <p style='color: #6B7280; font-size: 14px; margin-top: 10px;'>Flight Ticket Price Prediction System</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

@st.cache_resource
def load_model():
    return joblib.load("models/flight_price_radar.pkl")

model = load_model()

st.subheader("Flights✈️")
col1, col2, col3 = st.columns(3)

with col1:
    source_city = st.selectbox("From", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
with col2:
    destination_city = st.selectbox("To", ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
with col3:
    flight_date = st.date_input("Flight Date", min_value=date.today())

st.subheader("Flight Details")
col4, col5, col6 = st.columns(3)

with col4:
    airline = st.selectbox("Airline", ["Vistara", "Air_India", "Indigo", "GO_FIRST", "AirAsia", "SpiceJet"])
with col5:
    flight_class = st.selectbox("Class", ["Economy", "Business"])
with col6:
    stops = st.selectbox("Stops", ["0", "1", "2"])

st.markdown("<br>", unsafe_allow_html=True) 

if st.button("Predict Price", use_container_width=True, type="primary"):
    
    days_left = (flight_date - date.today()).days
    
    if source_city == destination_city:
        st.error("Departure and destination cities cannot be the same! Please update your route.")
    else:
        try:
            expected_cols = model.feature_names_in_
            
            df_input = pd.DataFrame(0, index=[0], columns=expected_cols)
            
            df_input['days_left'] = days_left
            df_input['duration'] = 2.5  
            df_input['class'] = 1 if flight_class == "Business" else 0
            df_input['stops'] = int(stops)
            
            airline_col = f"airline_{airline}"
            source_col = f"source_city_{source_city}"
            dest_col = f"destination_city_{destination_city}"
            
            if airline_col in df_input.columns:
                df_input[airline_col] = 1
            if source_col in df_input.columns:
                df_input[source_col] = 1
            if dest_col in df_input.columns:
                df_input[dest_col] = 1
                
            predicted_price = model.predict(df_input)
            
            st.success("Search Successful!")
            st.markdown(f"""
    <div style='max-width: 400px; margin: 20px auto; text-align: center; padding: 15px; border-radius: 10px; background-color: #d1fae5; border: 2px solid #10b981;'>
        <h4 style='color: #047857; margin: 0; padding-bottom: 5px;'>Estimated Ticket Price</h4>
        <h1 style='color: #065f46; font-size: 32px; margin: 0;'>{predicted_price[0]:,.2f} TL</h1>
        <p style='color: #047857; font-size: 13px; margin: 8px 0 0 0;'>This price is a prediction for <strong>{days_left} days</strong> ahead.</p>
    </div>
""", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"A critical error occurred: {e}")