
import streamlit as st
import requests

# fastapi server URL
API_URL = "http://localhost:8000/predict"

# configure Streamlit app
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered",
)

st.title("Clinical Heart Disease Predictor")
st.write("Enter the following clinical parameters to predict the likelihood of heart disease:")

with st.form("patient_form"):
    st.subheader("Patient Vitals")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
        thalach = st.number_input("Max Heart Rate (thalach)", min_value=50, max_value=250, value=150)
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
        
    with col2:
        # mapping dicts fro text to model values
        cp_map = {
            "Typical Angina": "1", 
            "Atypical Angina": "2", 
            "Non-anginal Pain": "3", 
            "Asymptomatic": "4"
        }
        restecg_map = {
            "Normal": "0", 
            "ST-T wave abnormality": "1", 
            "Left ventricular hypertrophy": "2"
        }
        slope_map = {
            "Upsloping": "1", 
            "Flat": "2", 
            "Downsloping": "3"
        }
        thal_map = {
            "Normal": "3.0", 
            "Fixed Defect": "6.0", 
            "Reversable Defect": "7.0"
        }
        
        cp_choice = st.selectbox("Chest Pain Type", list(cp_map.keys()))
        restecg_choice = st.selectbox("Resting ECG", list(restecg_map.keys()))
        slope_choice = st.selectbox("ST Segment Slope", list(slope_map.keys()))
        thal_choice = st.selectbox("Thalassemia", list(thal_map.keys()))
        
    submit_button = st.form_submit_button("Run Diagnostics")
    
# handle event when user clicks button
if submit_button:
    
    payload = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "oldpeak": oldpeak,
        "cp": cp_map[cp_choice],
        "restecg": restecg_map[restecg_choice],
        "slope": slope_map[slope_choice],
        "thal": thal_map[thal_choice]
    }
    
    try:
        with st.spinner("Transmitting data to ML Backend..."):
            # send data to fastapi
            response = requests.post(API_URL, json = payload)
            
            if response.status_code == 200:
                result = response.json()
                
                st.divider()
                st.subheader("Diagnostic Results")
                
                # display results based on risk level
                risk_percent = result["risk_probability"] * 100
                
                if result["prediction_class"] == 1:
                    st.error(f"**{result['clinical_status']}** (Confidence: {risk_percent:.1f}%)")
                else:
                    st.success(f"**{result['clinical_status']}** (Confidence: {risk_percent:.1f}%)")
                
                st.info(f"System Message: {result['message']}")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
    
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the backend api")
            