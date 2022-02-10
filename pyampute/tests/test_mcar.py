"""Unittest for mcar_tests.py"""
import unittest
import pandas as pd

from pyampute.exploration.mcar_statistical_tests import MCARTest


# load test data
data_mar = pd.read_table("data/missingdata.csv", sep="\t")
data_mcar = pd.read_table("data/missingdata_mcar.csv", sep="\t")

significance_level = 0.05


class TestMCARTest(unittest.TestCase):
    """Test for MCAR."""

    def test_littles_mcar_test(self):
        self.assertFalse(MCARTest(method="littles")(data_mcar) < significance_level)
        self.assertTrue(MCARTest(method="littles")(data_mar) < significance_level)

    def test_mcar_t_tests(self):
        # Axis=None reduces in all dimensions
        # should fail to reject sometimes, since MCAR
        self.assertTrue(
            (MCARTest(method="ttest")(data_mcar) > significance_level).any(axis=None)
        )
        # reject all: missingness is not MCAR for any pair of vars
        self.assertFalse(
            (MCARTest(method="ttest")(data_mar) < significance_level).all(axis=None)
        )


if __name__ == "__main__":
    unittest.main()
