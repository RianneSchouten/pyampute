"""
===================
Simulation Pipeline
===================

Multivariate amputation can be used to evaluate the effect of missing values on the outcome of an
 analysis or experiment. In general, such an experiment is designed as follows:

1. Generate or import a complete dataset
2. Ampute the dataset
3. Impute the dataset
4. Compare the performance of a model between the dataset in step 1, 2 and 3.
We will demonstrate how that works.

%%
Author: Rianne Schouten <r.m.schouten@tue.nl>

A complete dataset
------------------

A complete dataset can synthetically be designed.
In that case, be aware to create a correlation structure
between features, since correlation ensures that the missing values differ from the observed values,
which again results in a change of performance of your model (see [1]_). Furthermore, a correlation
structure is needed for many useful imputation methods.

Another option is to work with a complete portion of an already incomplete dataset.
"""
from pyampute.ampute import MultivariateAmputation
from pyampute.exploration.md_patterns import mdPatterns
import numpy as np




m = 1000
n = 10
compl_dataset = np.random.randn(n, m)
"""
%%
Multivariate Amputation
-----------------------

With our multivariate amputation methodology, it is straightforward
to generate all sorts of missing data problems.We advise to evaluate the
performance of your model for different settings of the algorithm. For instance, compare
MCAR, MAR and MNAR missingness, or compare different missingness
proportions. An explanation of the input arguments
can be found in the [documentation]_ and a more thorough explanation in [this blogpost]_.
The default settings generate 1 patterns with MAR missingness
for 50% of the rows. The incomplete dataset
can be explored using the mdPatterns class.
"""
ma = MultivariateAmputation()
incompl_data = ma.fit_transform(compl_dataset)
mdp = mdPatterns()
patterns = mdp._get_patterns(incompl_data)
"""
%%
Imputation
----------

Imputation can easily be done using existing methods.
 Because we make use of sklearn's TransformerMixin,
it is easy to combine amputation and imputation in one pipeline.

here some code that shows pipeline

%%
Evaluation
----------
As an example, here we demonstrate how you can evaluate the effect of missing
values on estimating the mean of a variable.
here some code that compares the mean under 1, 2 and 3, and shows differences for MCAR and MAR,
and differences for SimpleImputer and IterativeImputer.

%%
References
----------

.. [1] `Generating missing values ...
       <https://www.tandfonline.com/doi/full/10.1080/00949655.2018.1491577>`_,
       Rianne M. Schouten, Peter Lugtig & Gerko Vink, etc.

"""
