# Heart Disease Risk Prediction App

## Overview

This project is a Streamlit web application that allows users to explore a heart disease dataset and predict their individual risk of heart disease based on various health factors. The application demonstrates an end-to-end data science workflow, including data loading, preprocessing, exploratory data analysis (EDA), model training, and deployment using Streamlit.

## Key Features

- **Interactive Streamlit Interface:** A user-friendly web app built with Streamlit.
- **Data Exploration:** Allows users to view data summaries, distributions, and correlations.
- **Risk Prediction:** Predicts heart disease risk based on user-provided health information.
- **Machine Learning Model:** Utilizes a trained Random Forest Classifier to make predictions.
- **Clear Code Structure:** Well-organized code with modular functions for maintainability.

## Project Structure

heart_disease_app/
├── data/
│ └── raw/
│ └── heart_disease.csv # Data file
│ └── processed/
│ └── processed_data.csv
├── src/
│ ├── data_loader.py # Loads and preprocesses data
│ ├── eda.py # Exploratory data analysis functions
│ ├── model.py # Defines, trains, and loads the ML model
│ ├── predict.py # Functions for making predictions
│ ├── utils.py # Utility functions (e.g., for encoding)
│ ├── app.py # Streamlit app entry point
├── models/
│ └── heart_disease_model.pkl # Saved trained model
├── notebooks/
│ └── exploratory_analysis.ipynb # Jupyter Notebook for EDA (optional)
├── requirements.txt
├── README.md
├── .gitignore

## How to Run the App

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd heart_disease_app
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Place the dataset:**
    Download the `heart_disease.csv` dataset. Place it inside the `data/raw/` directory.

5.  **Run the `data_loader.py` script to preprocess the data**
    ```bash
    python src/data_loader.py
    ```
6.  **Run the Streamlit app:**

    ```bash
    streamlit run src/app.py
    ```

    The app will open in your web browser.

## Data Source

- [Heart Disease Dataset](The link for the dataset) manually put in the `data/raw/` directory.

## Model

- **Model:** Random Forest Classifier
- **Training:** The model is trained using scikit-learn on the provided dataset. See `src/model.py` for details.
- **Evaluation:** The model's performance is evaluated using accuracy and classification report.

## Future Work

- Implement other machine learning models (e.g., Logistic Regression, XGBoost).
- Add model interpretability techniques (e.g., feature importance plots).
- Deploy the app using a cloud platform (e.g., Heroku, AWS).
- Implement more sophisticated data preprocessing techniques.
- Incorporate user feedback to improve the app's usability.
