"""
===========================================
Amputing with a custom probability function
===========================================
"""

# %%
"""
Create complete data.
"""
from pyampute import MultivariateAmputation
from pyampute import ArrayLike
import numpy as np



n = 10000
X = np.random.randn(n, 2)

# %%
"""
Define custom probability function.
"""
# purely for demonstrative type hints


# Must produce values between 0 and 1
def min_max_scale(X: ArrayLike) -> ArrayLike:
    X_abs = np.abs(X)
    return (X_abs - X_abs.min()) / (X_abs.max() - X_abs.min())


# %%
"""
Define some patterns.
Include the custom score to probability function in whichever pattern(s) you desire.
"""
my_incomplete_vars = [np.array([0]), np.array([1]), np.array([1])]
my_freqs = np.array((0.3, 0.2, 0.5))
my_weights = [np.array([4, 1]), np.array([0, 1]), np.array([1, 0])]
my_score_to_probability_funcs = [min_max_scale, "sigmoid-right", "sigmoid-right"]
my_prop = 0.3

patterns = [
    {
        "incomplete_vars": incomplete_vars,
        "freq": freq,
        "weights": weights,
        "score_to_probability_func": score_to_probability_func,
    }
    for incomplete_vars, freq, weights, score_to_probability_func in zip(
        my_incomplete_vars, my_freqs, my_weights, my_score_to_probability_funcs
    )
]

# %%
"""
Run ampute.
"""

ma = MultivariateAmputation(prop=my_prop, patterns=patterns)
incomplete_data = ma.fit_transform(X)

# %%
