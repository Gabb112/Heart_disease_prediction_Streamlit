import pandas as pd
import numpy as np


def make_prediction(model, form_data):
    """
    Makes a prediction using the loaded model and user-provided data from the Streamlit form.

    Args:
        model: The loaded machine learning model.
        form_data (dict): A dictionary containing the user's input from the Streamlit form.

    Returns:
        int: The predicted heart disease status (0 or 1), or None if an error occurs.
    """
    try:
        # Convert the form data to a DataFrame (required by scikit-learn)
        input_df = pd.DataFrame([form_data])

        # Ensure the input data has the same columns as the training data (order matters!)
        # Assuming the model was trained on all columns except 'Heart Disease Status'
        # and the columns are properly ordered in the form_data dictionary

        # Make the prediction
        prediction = model.predict(input_df)

        return prediction[0]  # Return the single predicted value
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def interpret_prediction(prediction):
    """
    Provides a user-friendly interpretation of the prediction.

    Args:
        prediction (int): The predicted heart disease status (0 or 1).

    Returns:
        str: A human-readable interpretation of the prediction.
    """
    if prediction == 1:
        return "The model predicts a higher risk of heart disease.  It is recommended to consult with a healthcare professional for further evaluation."
    elif prediction == 0:
        return "The model predicts a lower risk of heart disease. However, this is not a guarantee, and maintaining a healthy lifestyle is still important."
    else:
        return "Unable to provide a prediction."


# Example usage (for testing)
if __name__ == "__main__":
    # This is just to run the function, not required in the app
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    import os

    # Create a dummy model and save it
    class DummyModel:
        def predict(self, data):
            # Always predict 0 for testing
            return [0]

    dummy_model = DummyModel()

    # Create a dummy form data
    dummy_form_data = {
        "Age": 50,
        "Gender": 1,
        "Blood Pressure": 120,
        "Cholesterol Level": 200,
        "Exercise Habits": 2,
        "Smoking": 0,
        "Family Heart Disease": 0,
        "Diabetes": 0,
        "BMI": 25.0,
        "High Blood Pressure": 0,
        "Low HDL Cholesterol": 0,
        "High LDL Cholesterol": 0,
        "Alcohol Consumption": 0,
        "Stress Level": 1,
        "Sleep Hours": 7,
        "Sugar Consumption": 1,
        "Triglyceride Level": 150,
        "Fasting Blood Sugar": 90,
        "CRP Level": 2.0,
        "Homocysteine Level": 10.0,
    }

    # Make a prediction
    prediction = make_prediction(dummy_model, dummy_form_data)

    # Interpret the prediction
    interpretation = interpret_prediction(prediction)

    print("Prediction:", prediction)
    print("Interpretation:", interpretation)
