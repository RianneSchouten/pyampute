# test_ampute.py

import pytest
import numpy as np
from ampute import MultivariateAmputation

X = np.random.randn(100, 2)

# define some arguments
my_patterns = np.matrix('1 0; 0 1; 0 1')
my_freqs = np.array((0.3, 0.2, 0.5))
my_weights = np.matrix('4 1; 0 1; 1 0')
my_prop = 0.3

# try run ampute
ma = MultivariateAmputation(prop=my_prop, patterns=my_patterns, freqs=my_freqs, weights=my_weights)
incomplete_data = ma.fit_transform(X)
print(incomplete_data)


# test that all mechanisms work
def test_mechanisms():

	# create complete data
	n = 1000
	X = np.random.randn(n, 2)

	for mechanism in ['MAR', 'MNAR', 'MCAR']:

		current_mechanisms = np.repeat(mechanism, 2)
		ma = MultivariateAmputation(mechanisms=current_mechanisms)
		incomplete_data = ma.fit_transform(X)
		assert incomplete_data.shape == X.shape

		count_missing_values_per_column = np.sum(np.isnan(incomplete_data), axis=0)	
		assert np.all(count_missing_values_per_column > (0.4 * 0.5 * n)) # expect: around 250
		assert np.sum(count_missing_values_per_column) > (0.4 * n) # expect: around 500

		# check if it also works if len(mechanisms) = 1
		ma = MultivariateAmputation(mechanisms=mechanism)
		incomplete_data = ma.fit_transform(X)
		assert np.all(count_missing_values_per_column > (0.4 * 0.5 * n)) # expect: around 250
		assert np.sum(count_missing_values_per_column) > (0.4 * n) # expect: around 500


# test one specific situation
def test_specific_situation():

	# create complete data
	n = 10000
	X = np.random.randn(n, 2)

	# define some arguments
	my_patterns = np.matrix('1 0; 0 1; 0 1')
	my_freqs = np.array((0.3, 0.2, 0.5))
	my_weights = np.matrix('4 1; 0 1; 1 0')
	my_prop = 0.3

	# run ampute
	ma = MultivariateAmputation(prop=my_prop, patterns=my_patterns, freqs=my_freqs, weights=my_weights)
	incomplete_data = ma.fit_transform(X)
	print(incomplete_data)
	assert incomplete_data.shape == X.shape

	#print(np.sum(np.sum(np.isnan(incomplete_data), axis=0))) # expect: around 3000
	#print(np.sum(np.isnan(incomplete_data), axis=0)[0]) # expect: around 2100
	#print(np.sum(np.isnan(incomplete_data), axis=0)[1]) # expect: around 900

	assert np.absolute((my_prop * len(X)) - np.sum(np.sum(np.isnan(incomplete_data), axis=0))) < (0.01 * n)
	assert np.absolute((0.7 * my_prop * len(X)) - np.sum(np.isnan(incomplete_data), axis=0)[0]) < (0.02 * n)
	assert np.absolute((0.3 * my_prop * len(X)) - np.sum(np.isnan(incomplete_data), axis=0)[1]) < (0.02 * n)

