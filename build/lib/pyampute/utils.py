""" Utils mainly to write code agnostic to numpy or pandas.  """
# Author: Davina Zamanzadeh <davzaman@gmail.com>

from typing import List, Optional, Union
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from os.path import join

ArrayLike = Union[pd.Series, np.array, List]
Matrix = Union[pd.DataFrame, np.ndarray]

LOOKUP_TABLE_PATH = join("data", "shift_lookup.csv")


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
    X: Union[ArrayLike, Matrix], vars_to_enforce: Optional[np.ndarray] = None,
) -> Matrix:
    """
    Vars to enforce could be column names (str), indices (int), or a mask (bool).
    """
    if isinstance(X, np.ndarray):
        X = np.array(list(map(pd.to_numeric, X)))
        all_nan_cols = np.isnan(X).all(axis=0)
        X = X[:, ~all_nan_cols]
    else:  # pd_df, or native python array
        # enforce pd df if native python list
        X = pd.DataFrame(X)
        if vars_to_enforce is not None:
            # if integer indices or boolean mask, grab column names
            if not isinstance(vars_to_enforce[0], str):
                vars_to_enforce = X.columns[vars_to_enforce]

            X[vars_to_enforce] = (
                X[vars_to_enforce]
                .apply(pd.to_numeric, errors="coerce")
                .dropna(axis=1, how="all")
            )
        else:
            X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    return X
