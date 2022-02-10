import numpy as np
from pyampute.ampute import MultivariateAmputation
from pyampute.exploration.md_patterns import mdPatterns
import pandas as pd
import matplotlib as m
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pyampute.exploration.mcar_statistical_tests import MCARTest

data_mcar = pd.read_table("data/missingdata_mcar.csv", sep="\t")

mt = MCARTest(method='littles')
out = mt.littles_mcar_test(data_mcar)
print('out:', out)

out = mt.mcar_t_tests(data_mcar)
print('out2:', out)

out = MCARTest()(data_mcar)
print('out:', out)

class CustomEstimator(BaseEstimator):

    def __init__(self):
        super().__init__()

    def fit(self, X):
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

print(compl_dataset.shape)
print(np.corrcoef(compl_dataset.T))

ma = MultivariateAmputation(prop=0.8, patterns=[{'incomplete_vars': [0]}])
imp = SimpleImputer()
est = CustomEstimator()
#est = LinearRegression()

#ma.fit(compl_dataset)
#imp.fit(compl_dataset)
#est.fit(compl_dataset)

X, y = compl_dataset[:,:-1], compl_dataset[:,-1]
out = ma.fit(X)
print(out)
out2 = imp.fit(out)
print(out2)
out3 = est.fit(out2)
print(out3)

#incompl_dataset = ma.transform(compl_dataset)
#print(incompl_dataset[:,])

#imp_dataset = imp.transform(incompl_dataset)
#print(imp_dataset[:,])

#values = est.predict(imp_dataset)
#print(values)

incompl_dataset = ma.transform(X)
print(incompl_dataset[:,])
imp_dataset = imp.transform(incompl_dataset)
values = est.predict(imp_dataset)

sc = mean_squared_error(X[:,0], values)
print('without pipeline:', sc)

def my_custom_loss_func(y_true, y_pred):

    diff = np.abs(np.mean(y_true) - np.mean(y_pred))
    return diff

sc = my_custom_loss_func(X[:,0], values)
print('with custom function:', sc)

# with a pipeline
steps = [('amputation', MultivariateAmputation()), ('imputation', SimpleImputer()), ('estimator', CustomEstimator())]
pipe = Pipeline(steps)
print('pipe ', pipe)
pipe.fit(X=compl_dataset)
print(pipe)
pipe.transform(compl_dataset[:,:-1])

# with grid search
parameters = {'amputation_prop':[0.1, 0.3, 0.5, 0.7, 0.9]}

grid = GridSearchCV(estimator=pipe, param_grid=parameters, scoring='neg_mean_squared_error')
grid.fit(compl_dataset[:,:-1])

#print('score ', grid.score(compl_dataset[:,:-1], compl_dataset[:,-1]))

attrs = vars(grid)
#print(', '.join("%s: %s" % item for item in attrs.items()))

#score = make_scorer(my_custom_loss_func, greater_is_better=False)

#print(my_custom_loss_func(compl_dataset[:,0], est.predict(imp_dataset)))
#print(score(est,imp_dataset,compl_dataset[:,0]))




'''
n = 1000
m = 4
compl_dataset = np.random.randn(n, m)

mean = [5,5,5,5]
cor = 0.5
cov = [[1,cor,cor,cor],[cor,1,cor,cor,],[cor,cor,1,cor],[cor,cor,cor,1]]
compl_dataset = np.random.multivariate_normal(mean, cov, n)
print(compl_dataset.shape)
print(np.corrcoef(compl_dataset, rowvar=False))

ma = MultivariateAmputation(
            patterns = [
                {'incomplete_vars': [3], 'weights': [0,4,1,0]},
                {'incomplete_vars': [2]},
                {'incomplete_vars': [1,2], 'mechanism': "MNAR"},
                {'incomplete_vars': [1,2,3], 'weights': {0:-2,3:1}, 'mechanism': "MAR+MNAR"}
            ]
        )
#incompl_data = ma.fit_transform(compl_dataset)

ma = ma.fit(compl_dataset)
attrs = vars(ma)
#print(', '.join("%s: %s" % item for item in attrs.items()))

incompl_data = ma.transform(compl_dataset)
attrs = vars(ma)
#print(', '.join("%s: %s" % item for item in attrs.items()))
'''
'''
mdp = mdPatterns()
patterns = mdp.get_patterns(incompl_data)

np.mean(incompl_data)

imp = SimpleImputer()
X_imp_test = imp.fit_transform(incompl_data)

print(X_imp_test[:20,:])

print(np.apply_along_axis(np.bincount, 1, X_imp_test))
t = np.apply_along_axis(np.bincount, 1, X_imp_test)

X_train, X_test = train_test_split(compl_dataset, random_state=0)
ma = MultivariateAmputation(patterns = [{'incomplete_vars': [1]}], prop = 0.8)
ma = ma.fit(X_train)

mdp1 = mdPatterns()
patterns = mdp1.get_patterns(X_train)
print(patterns)

X_test_incomplete = ma.transform(X_test)
mdp2 = mdPatterns()
patterns = mdp2.get_patterns(X_test_incomplete)
print(patterns)

X_train, X_test = train_test_split(compl_dataset, random_state=1)
pipe = make_pipeline(MultivariateAmputation(patterns = [{'incomplete_vars': [1]}], prop = 0.8), SimpleImputer())

pipe.fit(X_train)
mdp1 = mdPatterns()
patterns = mdp1.get_patterns(X_train)
print(patterns)

print(X_train[:10,:])
print(X_test[:10,:])

X_new = pipe.transform(X_test)
mdp2 = mdPatterns()
patterns = mdp2.get_patterns(X_new)
print(patterns)

print(X_new[:10,:])

'''
