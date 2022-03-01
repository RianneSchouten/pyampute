import numpy as np
import pandas as pd
import unittest

from pyampute.utils import enforce_numeric


class TestEnforceNumeric(unittest.TestCase):
    def setUp(self) -> None:
        self.array = [
            [44, "15.1", 0, "0", 1, 0],
            [49, 57.2, 1, "0", 0, 1],
            [26, "26.3", 0, "0", 1, 0],
            [16, 73.4, 1, "1", 0, 0],
            [13, 56.5, 1, "0", 1, 0],
            [57, 29.6, 0, "1", 0, 0],
        ]
        self.numeric = [
            [44, 15.1, 0, 0, 1, 0],
            [49, 57.2, 1, 0, 0, 1],
            [26, 26.3, 0, 0, 1, 0],
            [16, 73.4, 1, 1, 0, 0],
            [13, 56.5, 1, 0, 1, 0],
            [57, 29.6, 0, 1, 0, 0],
        ]
        self.columns = ["age", "weight", "ismale", "fries_s", "fries_m", "fries_l"]
        self.cols_to_enforce = [1, 3]
        return super().setUp()

    def test_native_python(self):
        enforce_all = enforce_numeric(self.array)
        numeric = enforce_numeric(self.array, self.cols_to_enforce)
        correct = pd.DataFrame(self.numeric)

        self.assertTrue(enforce_all.equals(numeric))
        self.assertTrue(numeric.equals(correct))

    def test_numpy(self):
        array = np.array(self.array)
        enforce_all = enforce_numeric(array)
        numeric = enforce_numeric(array, self.cols_to_enforce)
        correct = np.array(self.numeric)

        self.assertTrue(np.array_equal(enforce_all, numeric))
        self.assertTrue(np.array_equal(numeric, correct))

    def test_pandas(self):
        array = pd.DataFrame(self.array, columns=self.columns)
        enforce_all = enforce_numeric(array)
        # Test passing indices of vars to enforce
        numeric_idx = enforce_numeric(array, self.cols_to_enforce)
        # Test passing col names of vars to enforce
        numeric_strname = enforce_numeric(array, array.columns[self.cols_to_enforce])
        correct = pd.DataFrame(self.numeric, columns=self.columns)
        # Test boolean mask for columns
        boolean_mask = [
            True if i in self.cols_to_enforce else False
            for i in range(len(self.columns))
        ]
        boolean_column = enforce_numeric(array, boolean_mask)

        self.assertTrue(enforce_all.equals(numeric_idx))
        self.assertTrue(numeric_strname.equals(numeric_idx))
        self.assertTrue(numeric_strname.equals(correct))
        self.assertTrue(boolean_column.equals(numeric_strname))


if __name__ == "__main__":
    unittest.main()
