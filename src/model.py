import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle
import os


class HeartDiseaseModel:
    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(random_state=random_state)
        self.random_state = random_state

    def train(self, X_train, y_train):
        try:
            self.model.fit(X_train, y_train)
            print("Model trained successfully.")
        except Exception as e:
            print(f"Error during model training: {e}")

    def evaluate(self, X_test, y_test):
        try:
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            print(f"Accuracy: {accuracy}")
            print(f"Classification Report:\n{report}")
            return accuracy, report
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            return None, None

    def save_model(self, filepath):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as file:
                pickle.dump(self.model, file)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filepath):
        try:
            with open(filepath, "rb") as file:
                self.model = pickle.load(file)
            print(f"Model loaded from {filepath}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self, data):
        try:
            prediction = self.model.predict(data)
            return prediction
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return None


# Example usage (for testing purposes)
if __name__ == "__main__":
    project_dir = os.getcwd()
    processed_data_path = os.path.join(
        project_dir, "data", "processed", "processed_data.csv"
    )
    model_path = os.path.join(project_dir, "models", "heart_disease_model.pkl")

    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        print("First run data_loader.py to create a processed_data.csv file")
        exit()

    X = df.drop("Heart Disease Status", axis=1)
    y = df["Heart Disease Status"]

    # Splitting train test here, because the data_loader does it just as a demo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = HeartDiseaseModel()
    model.train(X_train, y_train)
    accuracy, report = model.evaluate(X_test, y_test)
    model.save_model(model_path)

    # Load the model
    loaded_model = HeartDiseaseModel()
    loaded_model.load_model(model_path)

    # Make a prediction (example)
    sample_data = X_test.iloc[[0]]
    prediction = loaded_model.predict(sample_data)
    print("Sample Prediction:", prediction)
