"""
===============
A quick example
===============

Generating missing values in complete datasets can be done with :class:`~pyampute.ampute.MultivariateAmputation`. This is useful for understanding and evaluating the effect of missing values on the outcome of a model. 

:class:`~pyampute.ampute.MultivariateAmputation` is designed as an `sklearn`_ `TransformerMixin`_, to allow for easy integration in a `pipeline`_.

Here, we give a short demonstration. A more extensive example of designing simulation studies for evaluating the effect of missing values can be found in `this example`_. For people who are familiar with the implementation of multivariate amputation in R-function `ampute`_, `this blogpost`_ gives an overview of the similarities and differences with :class:`~pyampute.ampute.MultivariateAmputation`. Inspection of an incomplete dataset can be done with :class:`~pyampute.exploration.md_patterns.mdPatterns`.

Note that the amputation methodology itself is proposed in `Generating missing values for simulation purposes`_ and used in `The dance of the mechanisms`_.

.. _`sklearn`: https://scikit-learn.org/stable/index.html
.. _`TransformerMixin`: https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin
.. _`pipeline`: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
.. _`this example`: https://rianneschouten.github.io/pyampute/build/html/auto_examples/plot_simulation_pipeline.html
.. _`ampute`: https://rianneschouten.github.io/mice_ampute/vignette/ampute.html
.. _`this blogpost`: https://rianneschouten.github.io/pyampute/build/html/mapping.html
.. _`Generating missing values for simulation purposes`: https://www.tandfonline.com/doi/full/10.1080/00949655.2018.1491577
.. _`The Dance of the Mechanisms`: https://journals.sagepub.com/doi/full/10.1177/0049124118799376
"""

# Author: Rianne Schouten <https://rianneschouten.github.io/>
# Co-Author: Davina Zamanzadeh <https://davinaz.me/>

# %%
# Transforming one dataset
##########################
#
# Multivariate amputation of one dataset can directly be performed with ``fit_transform``. Inspection of an incomplete dataset can be done with :class:`~pyampute.exploration.md_patterns.mdPatterns`. By default, :class:`~pyampute.ampute.MultivariateAmputation` generates 1 pattern with MAR missingness in 50% of the data rows for 50% of the variables.
#

import numpy as np

from pyampute.ampute import MultivariateAmputation
from pyampute.exploration.md_patterns import mdPatterns

rng = np.random.RandomState(2022)

m = 1000
n = 10
X_compl = np.random.randn(m,n)

ma = MultivariateAmputation()
X_incompl = ma.fit_transform(X_compl)

mdp = mdPatterns()
patterns = mdp.get_patterns(X_incompl)

# %%
# A separate fit and transform
##############################
#
# Evaluation of the effect of missing values on the outcome of a prediction model is best done by performing the amputation on the train and test set separately.
#

from sklearn.model_selection import train_test_split

X_compl_train, X_compl_test = train_test_split(X_compl, random_state=2020)
ma = MultivariateAmputation()
ma.fit(X_compl_train)
X_incompl_test = ma.transform(X_compl_test)

mdp = mdPatterns()
patterns = mdp.get_patterns(X_incompl_test)

# %%
# Application in a pipeline
###########################
#
# Because :class:`~pyampute.ampute.MultivariateAmputation` is designed as a `TransformerMixin`_, it is easy to set up an `sklearn`_ `pipeline`_ to evaluate several combinations of amputation settings and imputation methods.
#
# .. _`sklearn`: https://scikit-learn.org/stable/index.html
# .. _`TransformerMixin`: https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin
# .. _`pipeline`: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
#

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

pipe = make_pipeline(MultivariateAmputation(), SimpleImputer())
pipe.fit(X_compl_train)

X_imp_test = pipe.transform(X_compl_test)

# %%
# By default, SimpleImputer imputes with the mean of the observed data. It is therefore like that we find the median in 50% of the rows (of the test set, which contains 25% of m) for 50% of the variables.

medians = np.nanmedian(X_imp_test, axis=0)
print(np.sum(X_imp_test == medians[None,:], axis=0))
