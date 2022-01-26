import numpy as np
import pandas as pd
import unittest

from pyampute.exploration.md_patterns import mdPatterns
from pyampute.ampute import MultivariateAmputation

class TestEnforceNumeric(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_incomplete_nparray(self):

        incomplete_X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
        incomplete_X = np.asarray(incomplete_X)

        mdp = mdPatterns()
        patterns = mdp.get_patterns(incomplete_X, show_plot=False, show_patterns=False)

        self.assertEqual(patterns.shape, (3,5))
        self.assertEqual(patterns.iloc[0,1:-1].sum(), 2)
        self.assertEqual(patterns.iloc[1,1:-1].sum(), 2)
        self.assertEqual(patterns.iloc[0:-1,-1].sum(), 2)

    def test_output_ma_as_input(self):

        # create complete data
        n = 1000
        X = np.random.randn(n, 2)

        ma = MultivariateAmputation(seed=2022)
        incomplete_X = ma.fit_transform(X)

        mdp = mdPatterns()
        patterns = mdp.get_patterns(incomplete_X, show_plot=False, show_patterns=False)

        self.assertEqual(patterns.shape, (3,4))
        self.assertEqual(patterns.iloc[0,1:-1].sum(), 2)
        self.assertEqual(patterns.iloc[0,-1], 0)
        self.assertEqual(patterns.iloc[0,0], 515)
        self.assertEqual(patterns.iloc[1,0], 485)

'''
    def test_pd_dataframes(self):

        nhanes2 = pd.read_csv("../../data/nhanes2.csv")
        mdp = mdPatterns()
        patterns = mdp.get_patterns(nhanes2, show_plot=False, show_patterns=False)

        self.assertEqual(patterns.shape, (6,6))
        self.assertEqual(patterns.iloc[1:-1,1:-1].values, [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]])
'''

if __name__ == "__main__":
    unittest.main()