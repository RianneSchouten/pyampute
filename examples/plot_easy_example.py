"""
=============
Example Usage
=============

This is a very easy example of how the MultivariateAmputation can be used. The input is always a complete dataset. After amputation, an incomplete dataset can be explored using the mdPatterns class.

To further understand why you would want to ampute your complete dataset, see this example [1]_. In [this blogpost]_ we elaborate on how the input arguments can be specified and how they will lead to different kinds of missing data problems. 
"""
import numpy as np
from pyampute.ampute import MultivariateAmputation
from pyampute.exploration.md_patterns import mdPatterns

m = 1000
n = 10
compl_dataset = np.random.randn(n, m)

ma = MultivariateAmputation()
incompl_data = ma.fit_transform(compl_dataset)

mdp = mdPatterns()
patterns = mdp._get_patterns(incompl_data)