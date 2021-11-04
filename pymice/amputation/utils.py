""" Utils mainly to write code agnostic to numpy or pandas.  """
# Author: Davina Zamanzadeh <davzaman@gmail.com>

from typing import Callable, List, Union
import logging
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

ArrayLike = Union[pd.Series, np.array, List]
Matrix = Union[pd.DataFrame, np.ndarray]


def setup_logging(log_filename: str = "output.log"):
    # Ref: https://stackoverflow.com/a/46098711/1888794
    # prints to console and saves to File.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )


def standardize_uppercase(input: str) -> str:
    """Standardize string to upper case."""
    return input.upper()


def shifted_probability_func(
    wss_standardized: ArrayLike,
    shift_amount: float,
    probability_func: Union[str, Callable[[ArrayLike], ArrayLike]],
) -> ArrayLike:
    """
    Applies shifted custom function or sigmoid (with cutoff) to
        standardized weighted sum scores to convert to probability.
    String: sigmoid-
        Right: Regular sigmoid pushes larger values to have high probability,
        Left: To flip regular sigmoid across y axis, make input negative.
            This pushes smaller values to have high probability.
        We apply similar tricks for mid and tail, shifting appropriately.
    """

    if isinstance(probability_func, str):
        cutoff_transformations = {
            "SIGMOID-RIGHT": lambda wss_standardized, b: wss_standardized + b,
            "SIGMOID-LEFT": lambda wss_standardized, b: -wss_standardized + b,
            "SIGMOID-TAIL": lambda wss_standardized, b: (
                np.absolute(wss_standardized) - 0.75 + b
            ),
            "SIGMOID-MID": lambda wss_standardized, b: (
                -np.absolute(wss_standardized) + 0.75 + b
            ),
        }

        return sigmoid(
            cutoff_transformations[probability_func](wss_standardized, shift_amount)
        )
    return probability_func(wss_standardized) + shift_amount


def sigmoid(X: ArrayLike) -> ArrayLike:
    return 1 / (1 + np.exp(-X))


def isnan(X: Matrix):
    # ref: https://stackoverflow.com/a/29530601/1888794
    if isinstance(X, pd.DataFrame):
        return X.isnull().values
    # else np.ndarray
    return np.isnan(X)


def isin(X: Union[ArrayLike, Matrix], list: ArrayLike) -> bool:
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.isin(list)
    return np.isin(X, list)


def is_numeric(X: Union[ArrayLike, Matrix]) -> bool:
    if isinstance(X, pd.DataFrame):
        return is_numeric_dtype(X.values)

    return is_numeric_dtype(X)


def enforce_numeric(X: Union[ArrayLike, Matrix]) -> Matrix:
    if isinstance(X, pd.DataFrame):
        X = X.apply(pd.to_numeric, errors="coerce")
    elif isinstance(X, np.ndarray):
        X = np.array(list(map(pd.to_numeric, X)))
    else:
        X = pd.to_numeric(X, errors="coerce")

    return X


def missingness_profile(X: Matrix):
    nans = isnan(X)

    def percentify(n: int, axis: int) -> float:
        """Take number and report as a percent of axis."""
        return n / X.shape[axis] * 100

    # By row
    entries_missing = nans.any(axis=1).sum()
    print(
        "Entries missing a value: "
        f"{entries_missing} ({percentify(entries_missing, 0)}%)"
    )

    # By column
    print("Features missing | %")
    feature_missing = nans.sum(axis=0)
    percent_feature_missing = percentify(feature_missing, 0)
    print(np.concatenate(feature_missing, percent_feature_missing))
