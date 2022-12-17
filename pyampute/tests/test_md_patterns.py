import numpy as np
import pandas as pd
import unittest

from pyampute.exploration.md_patterns import mdPatterns
from pyampute.ampute import MultivariateAmputation


class TestMdPatterns(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        try:
            self.nhanes2 = pd.read_csv("data/nhanes2.csv")
        except:
            print("CSV file failed to load.")

    def test_incomplete_nparray(self):

        # create incomplete dataset
        incomplete_X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
        incomplete_X = np.asarray(incomplete_X)

        # create patterns
        mdp = mdPatterns()
        patterns = mdp.get_patterns(incomplete_X, show_plot=False)

        self.assertEqual(patterns.shape, (4, 5))
        self.assertListEqual(
            patterns.loc["rows_no_missing"].values.tolist(), [0, 1, 1, 1, 0]
        )

        self.assertEqual(patterns.loc[1].values[1:-1].sum(), 2)
        self.assertEqual(patterns.loc[2].values[1:-1].sum(), 2)
        self.assertEqual(patterns.loc[[1, 2], "n_missing_values"].sum(), 2)

    def test_output_ma_as_input(self):

        # create complete dataset
        rng = np.random.default_rng(2022)
        n = 1000
        X = rng.standard_normal((n, 2))

        # ampute the dataset with a seed value
        ma = MultivariateAmputation(seed=2022)
        incomplete_X = ma.fit_transform(X)

        # create patterns
        mdp = mdPatterns()
        patterns = mdp.get_patterns(incomplete_X, show_plot=False)

        self.assertEqual(patterns.shape, (3, 4))
        self.assertEqual(patterns.loc["rows_no_missing"].values[1:-1].sum(), 2)
        self.assertEqual(patterns.loc["rows_no_missing", "n_missing_values"], 0)
        self.assertEqual(patterns.loc["rows_no_missing", "row_count"], 508)
        self.assertEqual(patterns.loc[1, "row_count"], 492)

        # self.assertEqual(patterns.iloc[0, 1:-1].sum(), 2)
        # self.assertEqual(patterns.iloc[0, -1], 0)
        # self.assertEqual(patterns.iloc[0, 0], 489)
        # self.assertEqual(patterns.iloc[1, 0], 511)

    def test_pd_dataframes(self):

        mdp = mdPatterns()
        patterns = mdp.get_patterns(self.nhanes2, show_plot=False)

        self.assertEqual(patterns.shape, (6, 6))
        self.assertListEqual(
            patterns.iloc[1:-1, 1:-1].values.tolist(),
            [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]],
        )

    def test_proportions(self):

        mdp = mdPatterns()
        patterns = mdp.get_patterns(
            self.nhanes2, count_or_proportion="proportion", show_plot=False
        )

        self.assertEqual(patterns["row_prop"].values[:-1].astype(float).sum(), 1.0)
        self.assertListEqual(
            patterns.loc["n_missing_values_per_col"].values[1:].astype(float).tolist(),
            [0.0, 0.32, 0.36, 0.4, 0.27],
        )

        # self.assertEqual(patterns.iloc[0:-1, 0].astype(float).values.sum(), 1.0)
        # self.assertListEqual(
        #    patterns.iloc[-1, 1:].values.tolist(), [0.0, 0.32, 0.36, 0.4, 0.27]
        # )


if __name__ == "__main__":
    unittest.main()
