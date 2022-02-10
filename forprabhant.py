import numpy as np
import pandas as pd
from pyampute.ampute import MultivariateAmputation
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator 
from sklearn.linear_model import LinearRegression

class CustomEstimator(BaseEstimator):

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        self.X = X
        
        return self

    def predict(self, X):
        values_used_for_score = X[:,0]
		
        return values_used_for_score

m = 4
n = 1000

mean = np.repeat(5,m)
cor = 0.5
cov = np.identity(m)
cov[cov == 0] = cor
compl_dataset = np.random.multivariate_normal(mean, cov, n)

print(compl_dataset[:10,:])

steps = [('amputation', MultivariateAmputation()), ('imputation', SimpleImputer()), ('estimator', LinearRegression())]
pipe = Pipeline(steps)
print('pipe ', pipe)
pipe.fit(compl_dataset[:,:-1],compl_dataset[:,-1])
print(pipe)
#pipe.transform(compl_dataset[:,:-1])
vals = pipe.predict(compl_dataset[:,:-1])

steps = [('amputation', MultivariateAmputation()), ('imputation', SimpleImputer()), ('estimator', CustomEstimator())]
pipe = Pipeline(steps)
print('pipe ', pipe)
pipe.fit(compl_dataset[:,:-1],compl_dataset[:,-1])
print(pipe)
#pipe.transform(compl_dataset[:,:-1])
vals = pipe.predict(compl_dataset[:,:-1])
print(vals)

# grid
steps = [('amputation', MultivariateAmputation()), ('imputation', SimpleImputer()), ('estimator', LinearRegression())]
pipe = Pipeline(steps)

parameters = dict(amputation__prop=[0.1, 0.3, 0.5, 0.7, 0.9], amputation__mechanism = ["MCAR", "MAR", "MNAR"])
#parameters = [{'imputation_strategy':["mean","median"]}]

grid = GridSearchCV(estimator=pipe, param_grid=parameters, scoring='neg_mean_squared_error')
grid.fit(compl_dataset[:,:-1],compl_dataset[:,-1])

print(grid)


'''
steps = [('amputation', MultivariateAmputation()), ('imputation', SimpleImputer()), ('estimator', CustomEstimator())]
pipe = Pipeline(steps)

parameters = [{'amputation_prop':[0.1, 0.3, 0.5, 0.7, 0.9]}]

grid = GridSearchCV(estimator=pipe, param_grid=parameters, scoring='neg_mean_squared_error')
grid.fit(compl_dataset)
'''