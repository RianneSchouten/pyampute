""" Utils mainly to write code agnostic to numpy or pandas.  """
# Author: Davina Zamanzadeh <davzaman@gmail.com>

from typing import List, Optional, Union
import logging
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from os import getcwd
from os.path import join

ArrayLike = Union[pd.Series, np.array, List]
Matrix = Union[pd.DataFrame, np.ndarray]

LOOKUP_TABLE_PATH = join(getcwd(), "data", "shift_lookup.csv")


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


def sigmoid(X: ArrayLike) -> ArrayLike:
    return 1 / (1 + np.exp(-X))


def isin(X: Union[ArrayLike, Matrix], list: ArrayLike) -> bool:
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.isin(list)
    return np.isin(X, list)


def is_numeric(X: Union[ArrayLike, Matrix]) -> bool:
    if isinstance(X, pd.DataFrame):
        return is_numeric_dtype(X.values)

    return is_numeric_dtype(X)


def enforce_numeric(
    X: Union[ArrayLike, Matrix], vars_to_enforce: Optional[List[Union[str, int]]] = None
) -> Matrix:
    if isinstance(X, np.ndarray):
        X = np.array(list(map(pd.to_numeric, X)))
        all_nan_cols = np.isnan(X).all(axis=0)
        X = X[:, ~all_nan_cols]
    else:  # pd_df, or native python array
        # enforce pd df if native python list
        X = pd.DataFrame(X)
        if vars_to_enforce is not None:
            X[vars_to_enforce] = (
                X[vars_to_enforce]
                .apply(pd.to_numeric, errors="coerce")
                .dropna(axis=1, how="all")
            )
        else:
            X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    return X


'''
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
    feature_missing = pd.Series(nans.sum(axis=0))
    percent_feature_missing = pd.Series(percentify(feature_missing, 0))
    print(
        pd.concat(
            [feature_missing, percent_feature_missing],
            axis=1,
            keys=["Num Features Missing", "%"],
        )
    )
'''
