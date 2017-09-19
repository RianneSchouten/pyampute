"""Unittest for mcar_tests.py"""
import unittest
import pandas as pd
from pymice.exploration.mcar_tests import *

# load test data
data_mar = pd.read_table('data/missingdata.csv', sep='\t')
data_mcar = pd.read_table('data/missingdata_mcar.csv', sep='\t')

class_data_mcar = McarTests(data=data_mcar)
class_data_mar = McarTests(data=data_mar)

class TestMcarTests(unittest.TestCase):
    """Test for McarTests"""
    def test_mcar_test(self):
        """Test whether mcar_test() returns correct output"""
        self.assertFalse(class_data_mcar.mcar_test() < 0.05)
        self.assertTrue(class_data_mar.mcar_test() < 0.05)

    def test_mcar_t_tests(self):
        """Test whether mcar_t_tests() returns correct output"""
        self.assertTrue(class_data_mcar.mcar_t_tests().any().any())
        self.assertFalse(class_data_mar.mcar_t_tests().any().any())

if __name__ == '__main__':
    unittest.main()
