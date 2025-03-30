import numpy as np
import pandas as pd


def std_outlier_removal(df, cols, threshold=3):
    """
    Remove outliers from the dataframe based on the standard deviation method.

    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe
    cols : list
        List of column names to check for outliers
    threshold : float, optional (default=3)
        Number of standard deviations from the mean to identify outliers

    Returns:
    --------
    pandas DataFrame
        Dataframe with outliers removed
    """
    df_clean = df.copy()

    for col in cols:
        if col not in df_clean.columns:
            continue

        # Calculate the mean and standard deviation
        mean = df_clean[col].mean()
        std = df_clean[col].std()

        # Compute the threshold for outlier detection
        computed_threshold = threshold * std

        # Create a mask for non-outlier values
        mask = np.abs(df_clean[col] - mean) <= computed_threshold

        # Keep only non-outlier values
        df_clean = df_clean[mask]

    return df_clean
