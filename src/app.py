import streamlit as st
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.eda import EDA
from src.model import HeartDiseaseModel
from src.predict import make_prediction, interpret_prediction


def main():
    st.title("Heart Disease Risk Prediction App")

    # 1. Data Loading and EDA Section
    st.header("Data Exploration and Analysis")
    project_dir = os.getcwd()
    processed_data_path = os.path.join(
        project_dir, "data", "processed", "processed_data.csv"
    )

    try:
        df = pd.read_csv(processed_data_path)
        st.write("Data loaded successfully! Displaying first 5 rows:")
        st.dataframe(df.head())

        # EDA functionality
        eda = EDA(df)
        st.subheader("Exploratory Data Analysis")

        # Choose what to show
        eda_options = st.multiselect(
            "Select EDA Options",
            [
                "Basic Statistics",
                "Distributions",
                "Correlation Matrix",
                "Target Distribution",
            ],
            default=["Basic Statistics", "Distributions"],
        )

        if "Basic Statistics" in eda_options:
            eda.show_basic_stats()
        if "Distributions" in eda_options:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
            eda.plot_distributions(numeric_cols, categorical_cols)
        if "Correlation Matrix" in eda_options:
            eda.plot_correlation_matrix()
        if "Target Distribution" in eda_options:
            target_col = "Heart Disease Status"
            eda.plot_target_distribution(target_col)

    except FileNotFoundError:
        st.error(
            "Please run data_loader.py first to create the processed_data.csv file."
        )
        return  # Exit the app

    # 2. Prediction Section
    st.header("Heart Disease Risk Prediction")

    # Create a form for user input
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=18, max_value=120, value=50)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        blood_pressure = st.number_input(
            "Blood Pressure (Systolic)", min_value=80, max_value=220, value=120
        )
        cholesterol_level = st.number_input(
            "Cholesterol Level", min_value=50, max_value=500, value=200
        )
        exercise_habits = st.selectbox(
            "Exercise Habits", options=["Low", "Medium", "High"]
        )
        smoking = st.selectbox("Smoking", options=["Yes", "No"])
        family_history = st.selectbox(
            "Family History of Heart Disease", options=["Yes", "No"]
        )
        diabetes = st.selectbox("Diabetes", options=["Yes", "No"])
        bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0)
        high_blood_pressure = st.selectbox("High Blood Pressure", options=["Yes", "No"])
        low_hdl = st.selectbox("Low HDL Cholesterol", options=["Yes", "No"])
        high_ldl = st.selectbox("High LDL Cholesterol", options=["Yes", "No"])
        alcohol_consumption = st.selectbox(
            "Alcohol Consumption", options=["None", "Low", "Medium", "High"]
        )
        stress_level = st.selectbox("Stress Level", options=["Low", "Medium", "High"])
        sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=12, value=7)
        sugar_consumption = st.selectbox(
            "Sugar Consumption", options=["Low", "Medium", "High"]
        )
        triglyceride_level = st.number_input(
            "Triglyceride Level", min_value=50, max_value=500, value=150
        )
        fasting_blood_sugar = st.number_input(
            "Fasting Blood Sugar", min_value=50, max_value=200, value=90
        )
        crp_level = st.number_input(
            "CRP Level", min_value=0.0, max_value=10.0, value=2.0
        )
        homocysteine_level = st.number_input(
            "Homocysteine Level", min_value=5.0, max_value=30.0, value=10.0
        )

        submit_button = st.form_submit_button("Predict")

    if submit_button:
        # Prepare the input data for the model (same encoding as training)
        form_data = {
            "Age": age,
            "Gender": 1 if gender == "Male" else 0,
            "Blood Pressure": blood_pressure,
            "Cholesterol Level": cholesterol_level,
            "Exercise Habits": ["Low", "Medium", "High"].index(exercise_habits),
            "Smoking": 1 if smoking == "Yes" else 0,
            "Family Heart Disease": 1 if family_history == "Yes" else 0,
            "Diabetes": 1 if diabetes == "Yes" else 0,
            "BMI": bmi,
            "High Blood Pressure": 1 if high_blood_pressure == "Yes" else 0,
            "Low HDL Cholesterol": 1 if low_hdl == "Yes" else 0,
            "High LDL Cholesterol": 1 if high_ldl == "Yes" else 0,
            "Alcohol Consumption": ["None", "Low", "Medium", "High"].index(
                alcohol_consumption
            ),
            "Stress Level": ["Low", "Medium", "High"].index(stress_level),
            "Sleep Hours": sleep_hours,
            "Sugar Consumption": ["Low", "Medium", "High"].index(sugar_consumption),
            "Triglyceride Level": triglyceride_level,
            "Fasting Blood Sugar": fasting_blood_sugar,
            "CRP Level": crp_level,
            "Homocysteine Level": homocysteine_level,
        }

        # Load the model
        model_path = os.path.join(project_dir, "models", "heart_disease_model.pkl")
        heart_model = HeartDiseaseModel()
        heart_model.load_model(model_path)

        if heart_model.model is not None:  # Check if the model is correctly loaded
            # Make a prediction
            prediction = make_prediction(heart_model.model, form_data)

            # Interpret the prediction
            interpretation = interpret_prediction(prediction)

            # Display the result
            st.subheader("Prediction Result")
            st.write(interpretation)
        else:
            st.error("The model has not been loaded correctly, please check the logs.")


if __name__ == "__main__":
    main()
