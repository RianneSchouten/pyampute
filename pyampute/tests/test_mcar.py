"""Unittest for mcar_tests.py"""
import unittest
import pandas as pd

from pyampute.exploration.mcar_statistical_tests import McarTest


# load test data
data_mar = pd.read_table("data/missingdata.csv", sep="\t")
data_mcar = pd.read_table("data/missingdata_mcar.csv", sep="\t")

significance_level = 0.05

# TODO: better explain the algorithms for each test / better var names


class TestMcarTests(unittest.TestCase):
    """Test for MCAR."""

    def test_littles_mcar_test(self):
        self.assertFalse(McarTest(method="littles")(data_mcar) < significance_level)
        self.assertTrue(McarTest(method="littles")(data_mar) < significance_level)

    def test_mcar_t_tests(self):
        # Axis=None reduces in all dimensions
        # should fail to reject sometimes, since MCAR
        self.assertTrue(
            (McarTest(method="ttest")(data_mcar) > significance_level).any(axis=None)
        )
        # reject all: missingness is not MCAR for any pair of vars
        self.assertFalse(
            (McarTest(method="ttest")(data_mar) < significance_level).all(axis=None)
        )


if __name__ == "__main__":
    unittest.main()