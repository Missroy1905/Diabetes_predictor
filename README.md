# 🩺 Diabetes Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)

## 📌 Overview
This project is an end-to-end Machine Learning web application designed to predict the likelihood of a patient having diabetes based on various diagnostic measurements. 

Developed as part of the **AI/ML Virtual Internship with InternPe**, this project bridges the gap between raw data and a user-friendly diagnostic tool by deploying a trained model into a live, interactive web interface.

🌐 **[Try the Live App Here](Insert_Your_Streamlit_Link_Here)**

## 🚀 Features
* **Interactive UI:** Built with Streamlit for a seamless and responsive user experience.
* **Instant Predictions:** Users can input health metrics (Glucose, BMI, Age, etc.) and receive immediate risk assessments.
* **Robust Preprocessing:** Utilizes `StandardScaler` to normalize real-world medical data for accurate model evaluation.
* **Comparative Analysis:** Evaluated both Support Vector Machines (SVM) and Logistic Regression, deploying the latter for its strong baseline accuracy and high interpretability in medical contexts.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Community Cloud

## 📊 The Dataset
The model is trained on the standard **PIMA Indians Diabetes Database**. It includes 8 medical predictor variables and one target variable (Outcome).
* Pregnancies
* Glucose Level
* Blood Pressure
* Skin Thickness
* Insulin Level
* BMI
* Diabetes Pedigree Function
* Age

## 💻 How to Run Locally

If you want to run this project on your own machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[Your_GitHub_Username]/[Your_Repo_Name].git
   cd [Your_Repo_Name]