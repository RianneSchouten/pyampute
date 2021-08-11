""" Utils mainly to write code agnostic to numpy or pandas.  """
# Author: Davina Zamanzadeh <davzaman@gmail.com>

from typing import List, Union
import pandas as pd
import numpy as np

ArrayLike = Union[pd.Series, np.array, List]
Matrix = Union[pd.DataFrame, np.ndarray]


def sigmoid(X: ArrayLike) -> ArrayLike:
    return 1 / (1 + np.exp(-X))


def contains_nan(X: Matrix) -> bool:
    # ref: https://stackoverflow.com/a/29530601/1888794
    if isinstance(X, pd.DataFrame):
        return X.isnull().values.any()
    # else np.ndarray
    return np.isnan(X).any()


def isin(X: Union[ArrayLike, Matrix], list: ArrayLike) -> bool:
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.isin(list)
    return np.isin(X, list)


def enforce_numeric(X: Union[ArrayLike, Matrix]) -> Matrix:
    if isinstance(X, pd.DataFrame):
        X = X.apply(pd.to_numeric, errors="coerce")
    elif isinstance(X, np.ndarray):
        X = np.array(list(map(pd.to_numeric, X)))
    else:
        X = pd.to_numeric(X, errors="coerce")

    return X
