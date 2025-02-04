import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categorical_features(df, categorical_cols):
    """
    Encodes categorical features using Label Encoding.

    Args:
        df (pd.DataFrame): The DataFrame to encode.
        categorical_cols (list): List of categorical column names.

    Returns:
        pd.DataFrame: The encoded DataFrame.
        dict: A dictionary mapping column names to their LabelEncoder objects.
    """
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def decode_categorical_features(df, encoders):
    """
    Decodes categorical features using the fitted LabelEncoders.

    Args:
        df (pd.DataFrame): The DataFrame to decode.
        encoders (dict): A dictionary mapping column names to their LabelEncoder objects.

    Returns:
        pd.DataFrame: The decoded DataFrame.
    """
    for col, le in encoders.items():
        df[col] = le.inverse_transform(df[col])
    return df
