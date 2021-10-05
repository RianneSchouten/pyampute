"""Unittest for mcar_tests.py"""
import unittest
import pandas as pd
import numpy as np
from math import pow
from scipy.stats import chi2, ttest_ind

from pymice.amputation.utils import Matrix

# load test data
data_mar = pd.read_table("data/missingdata.csv", sep="\t")
data_mcar = pd.read_table("data/missingdata_mcar.csv", sep="\t")

significance_level = 0.05

# TODO: better explain the algorithms for each test / better var names


class TestMcarTests(unittest.TestCase):
    """Test for MCAR."""

    def test_littles_mcar_test(self):
        self.assertFalse(self.littles_mcar_test(data_mcar) < significance_level)
        self.assertTrue(self.littles_mcar_test(data_mar) < significance_level)

    def test_mcar_t_tests(self):
        # Axis=None reduces in all dimensions
        # should fail to reject sometimes, since MCAR
        self.assertTrue(
            (self.mcar_t_tests(data_mcar) > significance_level).any(axis=None)
        )
        # reject all: missingness is not MCAR for any pair of vars
        self.assertFalse(
            (self.mcar_t_tests(data_mar) < significance_level).all(axis=None)
        )

    def littles_mcar_test(self, data: Matrix) -> float:
        """
        Implementation of Little's MCAR test
        Returns  the p_value, the outcome of a chi-square statistical test.
        Null hypothesis: "missingness mech of data is MCAR".
        """

        dataset = data.copy()
        vars = dataset.dtypes.index.values
        n_var = dataset.shape[1]

        # mean and covariance estimates
        # ideally, this is done with a maximum likelihood estimator
        gmean = dataset.mean()
        gcov = dataset.cov()

        # set up missing data patterns
        r = 1 * dataset.isnull()
        mdp = np.dot(r, list(map(lambda x: pow(2, x), range(n_var))))
        sorted_mdp = sorted(np.unique(mdp))
        n_pat = len(sorted_mdp)
        correct_mdp = list(map(lambda x: sorted_mdp.index(x), mdp))
        dataset["mdp"] = pd.Series(correct_mdp, index=dataset.index)

        # calculate statistic and df
        pj = 0
        d2 = 0
        for i in range(n_pat):
            dataset_temp = dataset.loc[dataset["mdp"] == i, vars]
            select_vars = ~dataset_temp.isnull().any()
            pj += np.sum(select_vars)
            select_vars = vars[select_vars]
            means = dataset_temp[select_vars].mean() - gmean[select_vars]
            select_cov = gcov.loc[select_vars, select_vars]
            mj = len(dataset_temp)
            parta = np.dot(
                means.T, np.linalg.solve(select_cov, np.identity(select_cov.shape[1]))
            )
            d2 += mj * (np.dot(parta, means))

        df = pj - n_var

        # perform test and save output
        p_value = 1 - chi2.cdf(d2, df)

        return p_value

    def mcar_t_tests(self, data: Matrix) -> Matrix:
        """
        MCAR t-tests for each pair of variables.

        Returns a matrix of p-values for each pair of variables.
        Null hypothesis: missingness in row variable is MCAR vs col variable.
    """
        dataset = data.copy()
        vars = dataset.dtypes.index.values
        mcar_matrix = pd.DataFrame(
            data=np.zeros(shape=(dataset.shape[1], dataset.shape[1])),
            columns=vars,
            index=vars,
        )

        for var in vars:
            for tvar in vars:
                part_one = dataset.loc[dataset[var].isnull(), tvar].dropna()
                part_two = dataset.loc[~dataset[var].isnull(), tvar].dropna()
                mcar_matrix.loc[var, tvar] = ttest_ind(
                    part_one, part_two, equal_var=False
                ).pvalue

        return mcar_matrix[mcar_matrix.notnull()]


if __name__ == "__main__":
    unittest.main()
