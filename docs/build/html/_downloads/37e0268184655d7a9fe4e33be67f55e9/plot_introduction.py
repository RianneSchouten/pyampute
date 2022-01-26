"""
========================================
Main title of example
========================================
Some explanation
"""

# %%
# Author: Rianne Schouten <r.m.schouten@tue.nl>
#
# Synthetic dataset
# -----------------
#
# Some explanation with a reference [1]_

import numpy as np

m = 1000
n = 10
com_dataset = np.zeros((m, n))

# %%
# Multivariate Amputation
# -----------------------
#
# Some explanation

from pyampute.ampute import MultivariateAmputation

ma = MultivariateAmputation()
incomplete_data = ma.fit_transform(com_dataset)

# %%

from pyampute.exploration.md_patterns import mdPatterns

mdp = mdPatterns()
plot = mdp._get_patterns(incomplete_data)

# %%
# References
# ----------
#
# .. [1] `Generating missing values ...
#        <https://www.tandfonline.com/doi/full/10.1080/00949655.2018.1491577>`_,
#        Rianne M. Schouten, Peter Lugtig & Gerko Vink, etc.
