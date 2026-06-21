
import streamlit as st 
import numpy as np
import pickle 

with open('lr_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('scaler.pkl','rb') as scaler_file:
    scaler = pickle.load(model_file)

st.title("Diabetes Prediction System")
st.write("Powered by Logistic Regression")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Number of Pregnancies", 0, 20, 1)
    blood_pressure = st.number_input("Blood Pressure", 0, 150, 72)
    insulin = st.number_input("Insulin Level", 0, 200, 120)
    dpf = st.number_input("Diabetes Pedigree Function", 0.000,  3.000, 0.471, format="%.3f")

with col2:
    glucose = st.number_input("Glucose Level", 0, 200, 120)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    bmi = st.number_input("BMI", 0.0, 70.0, 32.0, format="%.1f")
    age = st.number_input("Age", 21, 100, 33)

if st.button("Predict"):
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    std_data = scaler.transform(input_data_as_numpy_array)

    prediction = classifier.predict(std_data)

    st.markdown("---")
    if prediction[0] == 0:
        st.success('Prediction: The patient is likely NOT Diabetic')

    else:
        st.error('Prediction: The patient is likely diabetic.')