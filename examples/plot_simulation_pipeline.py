"""
====================================================
Evaluating missing values with a simulation pipeline
====================================================

Generating missing values in a complete dataset (we call this `amputation`) may seem like a bizarre thing to do. However, most people who work with all sorts of data will acknowledge that missing data is widespread and can be a severe issue for various types of analyses and models. In order to understand the effect of missing values and to know which missing data methods are appropriate in which situation, we perform simulation studies. And for that, we need amputation. 

With package ``pyampute``, we provide the multivariate amputation methodology proposed by `Schouten et al. (2018)`_. Because our :class:`~pyampute.ampute.MultivariateAmputation` class is built on an sklearn TransformerMixin, it is easy to integrate such evaluation in a larger experiment. Here, we will demonstrate how that works. 

.. _`Schouten et al. (2018)`: https://www.tandfonline.com/doi/full/10.1080/00949655.2018.1491577
"""

# Author: Rianne Schouten <https://rianneschouten.github.io/>

# %%
# General experimental setup
############################
#
# In general, evaluating the effect of missing values is done in four steps:
#
# 1. Generate or import a complete dataset
# 2. Ampute the dataset
# 3. Impute the dataset
# 4. Compare the performance of a model between the datasets in step 1, 2 and 3. 
#
# It is often wise to first inspect the effect of amputation (by comparing the datasets in steps 1 and 2) before comparing with the dataset in step 3. Let's get started.
#

# %%
# Complete datasets
###################
#
# A simulation starts with a complete dataset. Make sure that you use a dataset where variables are correlated with each other; otherwise it will not make sense to use weights in a MAR or MNAR mechanism (see `Schouten et al. (2021)`_ for a discussion on this topic). 
#
# .. `Schouten et al. (2021)`: https://journals.sagepub.com/doi/full/10.1177/0049124118799376

import numpy as np

m = 3
n = 1000

mean = np.repeat(5,m)
cor = 0.5
cov = np.identity(m)
cov[cov == 0] = cor
compl_dataset = np.random.multivariate_normal(mean, cov, n)

# %%
# Multivariate amputation
#########################
#
# Vary the parameters of the amputation procedure. As an example, we generate one missing data pattern with missing values in the first two variables. We vary the proportion of incomplete rows and the missingness mechanisms
#

parameters = {'amputation_prop': [0.1, 0.5, 0.9], 'amputation_patterns' : [{'incomplete_vars': [0,1], 'mechanism': "MCAR"}, {'incomplete_vars': [0,1], 'mechanism': "MAR"}, {'incomplete_vars': [0,1], 'mechanism': "MNAR"}]}

# %%
# Missing data methods
######################
#
# `SimpleImputer`_` is a univariate, single imputation method that is commonly used. However, in case of MCAR missingness, it distorts the relation with other variables, and in case of MAR and MNAR missingness it will not resolve issues with shifted variable distributions (see `Van Buuren (2018)`_) It may be better to use a method such as `IterativeImputer`_. 
#
# Yet, to demonstrate the working of a simulation pipeline, we will work with SimpleImputer for now.
#
# .. `SimpleImputer`: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
# .. `Van Buuren (2018)`: https://stefvanbuuren.name/fimd/
# .. `IterativeImputer`: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html

parameters['imputation_strategy'] = ["mean"]

# %%
# Evaluation
############
#
# How you wish to evaluate the appropriateness of a missing data method greatly depends on the goal of your model. When you develop a prediction or classification model, you may want to use a standard estimator and evaluation metric.
#
# Here, as an example, we evaluate using principles from statistical theory, where the goal is to find an unbiased and efficient estimate of a population parameter in a sample (that contains missing values). As an easy example, we evaluate an estimate of the mean of a variable, in this case the mean of the first variable. 
#
# Therefore, we set up an empty BaseEstimator that returns the values of the first variable. We then design a custom evaluation metric. Since we work with one complete dataset, the true population estimate is the mean of the variable in that dataset. Note, in case we would repeatedly sample a complete dataset, the values of the distribution would become the true population estimate. Here, that would be ``true_mean = 5``.
#

from sklearn.base import BaseEstimator

class CustomEstimator(BaseEstimator):

    def __init__(self):
        super().__init__()

    def fit(self, X):
        self.X = X
        
        return self

    def predict(self, X):
        values_first_variable = X[:,0]
		
        return values_first_variable

def my_evaluation_metric(y_true, y_pred):

    bias = np.abs(np.mean(y_true) - np.mean(y_pred))

    return bias

# %%
# Altogether
############
#
# We then create our pipeline, and run an exhaustive grid search to see the effect of various parameters on the bias of the mean of the first variable.
#

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from pyampute.ampute import MultivariateAmputation

steps = [('amputation', MultivariateAmputation()), ('imputation', SimpleImputer()), ('estimator', CustomEstimator())]
pipe = Pipeline(steps)
grid = GridSearchCV(estimator=pipe, param_grid=parameters, scoring=my_evaluation_metric)

grid.fit(compl_dataset)
grid.score(compl_dataset, compl_dataset[:,0])
pd.DataFrame(grid.cv_results_)
