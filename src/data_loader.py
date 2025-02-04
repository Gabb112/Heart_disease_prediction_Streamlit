import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            print("Data loaded successfully.")
            return self.df
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_data(self, test_size=0.2, random_state=42, save_processed_path=None):
        """
        Preprocesses the data, including:
            - Handling missing values (if any).
            - Encoding categorical features.
            - Scaling numerical features.
            - Splitting the data into training and testing sets.
        """
        if self.df is None:
            print("Error: Data not loaded. Call load_data() first.")
            return None, None, None, None

        # Handle missing values (replace with mean for numerical, mode for categorical)
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        # Encode categorical features
        for col in self.df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(
                self.df[col]
            )  # Store the fitted encoders if needed

        # Scale numerical features
        numerical_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        numerical_cols.remove("Heart Disease Status")  # Don't scale the target
        scaler = StandardScaler()
        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])

        # Split into training and testing sets
        X = self.df.drop("Heart Disease Status", axis=1)
        y = self.df["Heart Disease Status"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if save_processed_path:
            os.makedirs(os.path.dirname(save_processed_path), exist_ok=True)
            self.df.to_csv(save_processed_path, index=False)
            print(f"Preprocessed data saved to : {save_processed_path}")

        return X_train, X_test, y_train, y_test


# Example usage (for testing purposes)
if __name__ == "__main__":
    project_dir = os.getcwd()
    raw_data_path = os.path.join(
        project_dir, "data", "raw", "heart_disease.csv"
    )  # Modify the path

    data_loader = DataLoader(raw_data_path)
    df = data_loader.load_data()

    if df is not None:
        processed_data_path = os.path.join(
            project_dir, "data", "processed", "processed_data.csv"
        )
        X_train, X_test, y_train, y_test = data_loader.preprocess_data(
            save_processed_path=processed_data_path
        )

        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)
