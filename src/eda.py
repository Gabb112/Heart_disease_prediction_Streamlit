import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os


class EDA:
    def __init__(self, df):
        self.df = df

    def show_basic_stats(self):
        st.header("Basic Statistics")
        st.dataframe(self.df.describe())

    def plot_distributions(self, numeric_cols, categorical_cols):
        st.header("Distributions of Features")

        st.subheader("Numeric Features")
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(self.df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        st.subheader("Categorical Features")
        for col in categorical_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(x=self.df[col], ax=ax)
            ax.set_title(f"Countplot of {col}")
            st.pyplot(fig)

    def plot_correlation_matrix(self):
        st.header("Correlation Matrix")
        corr = self.df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    def plot_target_distribution(self, target_col):
        st.header("Distribution of Target Variable")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=self.df[target_col], ax=ax)
        ax.set_title(f"Distribution of {target_col}")
        st.pyplot(fig)


# Example usage (for testing purposes)
if __name__ == "__main__":
    project_dir = os.getcwd()
    processed_data_path = os.path.join(
        project_dir, "data", "processed", "processed_data.csv"
    )

    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        print("First run data_loader.py to create a processed_data.csv file")
        exit()

    eda = EDA(df)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    target_col = "Heart Disease Status"

    # Now we should integrate this inside the streamlit app
    eda.show_basic_stats()
    eda.plot_distributions(numeric_cols, categorical_cols)
    eda.plot_correlation_matrix()
    eda.plot_target_distribution(target_col)
