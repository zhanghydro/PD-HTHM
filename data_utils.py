import numpy as np
import pandas as pd
from typing import Tuple, Dict

def generate_train_val_test(
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
    test_set: pd.DataFrame,
    wrap_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits raw DataFrames into training, validation, and test sets with appropriate shapes for model input.

    Args:
        train_set: DataFrame for training set with features and target in last column.
        val_set: DataFrame for validation set (will be expanded to 3D tensor).
        test_set: DataFrame for test set (will be expanded to 3D tensor).
        wrap_length: Length of the sliding window for training data.

    Returns:
        Tuple of numpy arrays: train_x, train_y, val_x, val_y, test_x, test_y
    """
    train_x_np = train_set.iloc[:, :-1].values
    train_y_np = train_set.iloc[:, -1:].values

    wrap_number_train = (train_x_np.shape[0] - wrap_length) // 365 + 1
    train_x = np.empty((wrap_number_train, wrap_length, train_x_np.shape[1]))
    train_y = np.empty((wrap_number_train, wrap_length, train_y_np.shape[1]))

    for i in range(wrap_number_train):
        train_x[i] = train_x_np[i * 365:(i * 365 + wrap_length)]
        train_y[i] = train_y_np[i * 365:(i * 365 + wrap_length)]

    val_x = np.expand_dims(val_set.iloc[:, :-1].values, axis=0)
    val_y = np.expand_dims(val_set.iloc[:, -1:].values, axis=0)
    test_x = np.expand_dims(test_set.iloc[:, :-1].values, axis=0)
    test_y = np.expand_dims(test_set.iloc[:, -1:], axis=0)

    return train_x, train_y, val_x, val_y, test_x, test_y

def create_sliding_windows(
    data_array: np.ndarray,
    window_length: int
) -> np.ndarray:
    """
    Constructs sliding windows for SHAP or inference purposes.

    Args:
        data_array: 2D array of features [time, features].
        window_length: Window size.

    Returns:
        3D array of shape (num_windows, window_length, num_features)
    """
    num_time_steps = data_array.shape[0]
    num_windows = num_time_steps - window_length + 1
    windows = np.empty((num_windows, window_length, data_array.shape[1]))
    for i in range(num_windows):
        windows[i] = data_array[i:i + window_length]
    return windows

def split_dataset_by_date(
    data: pd.DataFrame,
    date_col: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset based on date ranges for train, validation and test.

    Args:
        data: Original DataFrame with datetime index or date column.
        date_col: Name of the date column.
        train_end: End date for training.
        val_start: Start date for validation.
        val_end: End date for validation.
        test_start: Start date for testing.
        test_end: End date for testing.

    Returns:
        Tuple of DataFrames: train, validation, test
    """
    data[date_col] = pd.to_datetime(data[date_col])
    train_data = data[data[date_col] <= train_end].copy()
    val_data = data[(data[date_col] >= val_start) & (data[date_col] <= val_end)].copy()
    test_data = data[(data[date_col] >= test_start) & (data[date_col] <= test_end)].copy()
    return train_data, val_data, test_data


def inverse_normalization(
    data: pd.DataFrame,
    norm_stats: Dict[str, Tuple[float, float]],
    method: str = "zscore"
) -> pd.DataFrame:
    """
    Reverts normalized features to original scale.

    Args:
        data: Normalized DataFrame.
        norm_stats: Dictionary of (mean, std) or (min, max) per feature.
        method: 'zscore' or 'minmax'.

    Returns:
        DataFrame with original scale.
    """
    data = data.copy()
    for col, (a, b) in norm_stats.items():
        if method == "zscore":
            data[col] = data[col] * b + a
        elif method == "minmax":
            data[col] = data[col] * (b - a) + a
    return data

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reports number of missing values in each column of a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with counts and percentage of missing values.
    """
    missing = df.isnull().sum()
    percent = 100 * missing / len(df)
    return pd.DataFrame({'Missing Count': missing, 'Missing %': percent})

def generate_lag_features(
    df: pd.DataFrame,
    target_cols: list,
    lags: list
) -> pd.DataFrame:
    """
    Generates lag features for specified columns.

    Args:
        df: Input DataFrame.
        target_cols: List of columns to lag.
        lags: List of lag steps (int).

    Returns:
        DataFrame with original and lagged features.
    """
    df = df.copy()
    for col in target_cols:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

def detect_outliers_iqr(
    data: pd.Series,
    threshold: float = 1.5
) -> pd.Series:
    """
    Detects outliers using the IQR method.

    Args:
        data: Input 1D pandas Series.
        threshold: Multiplier for IQR (typically 1.5).

    Returns:
        Boolean Series where True indicates an outlier.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return (data < lower_bound) | (data > upper_bound)