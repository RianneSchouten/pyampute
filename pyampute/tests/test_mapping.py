import numpy as np
import pandas as pd
import unittest

from pyampute.ampute import MultivariateAmputation
from pyampute.exploration.md_patterns import mdPatterns


class TestMapping(unittest.TestCase):
    """
    This class tests the example code in the blogpost "A mapping from R-function ampute to pyampute"
    """

    def setUp(self) -> None:
        super().setUp()
        self.n = 10000
        rng = np.random.default_rng()
        self.nhanes2_sim = rng.standard_normal((10000, 4))
        try:
            self.nhanes2_orig = pd.read_csv("data/nhanes2.csv")
        except:
            print("CSV file failed to load.")

    def test_patterns(self):

        mdp = mdPatterns()
        mypatterns = mdp.get_patterns(self.nhanes2_orig, show_plot=False)

        self.assertEqual(mypatterns.shape, (6, 6))
        self.assertListEqual(
            mypatterns.iloc[1:-1, 1:-1].values.tolist(),
            [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]],
        )

        ma = MultivariateAmputation(
            patterns=[
                {"incomplete_vars": [3]},
                {"incomplete_vars": [2]},
                {"incomplete_vars": [1, 2]},
                {"incomplete_vars": [1, 2, 3]},
            ]
        )
        nhanes2_incomplete = ma.fit_transform(self.nhanes2_sim)
        mdp = mdPatterns()
        mypatterns = mdp.get_patterns(nhanes2_incomplete, show_plot=False)

        self.assertEqual(mypatterns.shape, (6, 6))
        self.assertListEqual(
            mypatterns["n_missing_values"].values[:-1].astype(int).tolist(),
            [0, 1, 1, 2, 3],
        )

    def test_proportions(self):

        ma = MultivariateAmputation(
            patterns=[
                {"incomplete_vars": [3], "freq": 0.1},
                {"incomplete_vars": [2], "freq": 0.6},
                {"incomplete_vars": [1, 2], "freq": 0.2},
                {"incomplete_vars": [1, 2, 3], "freq": 0.1},
            ],
            prop=0.3,
        )

        nhanes2_incomplete = ma.fit_transform(self.nhanes2_sim)
        mdp = mdPatterns()
        mypatterns = mdp.get_patterns(nhanes2_incomplete, show_plot=False)

        self.assertListEqual(
            mypatterns.columns.values.tolist(),
            ["row_count", 0, 3, 1, 2, "n_missing_values"],
        )
        self.assertAlmostEqual(
            mypatterns.loc[1, "row_count"], 0.3 * 0.6 * self.n, delta=0.05 * self.n,
        )

    def test_mechanisms(self):

        ma = MultivariateAmputation(
            patterns=[
                {"incomplete_vars": [3], "mechanism": "MCAR"},
                {"incomplete_vars": [2]},
                {"incomplete_vars": [1, 2], "mechanism": "MNAR"},
                {"incomplete_vars": [1, 2, 3]},
            ]
        )

        nhanes2_incomplete = ma.fit_transform(self.nhanes2_sim)

        self.assertEqual(ma.patterns[0]["mechanism"], "MCAR")
        self.assertEqual(ma.patterns[2]["mechanism"], "MNAR")

        self.assertListEqual(ma.mechanisms.tolist(), ["MCAR", "MAR", "MNAR", "MAR"])

    def test_weights(self):

        ma = MultivariateAmputation(
            patterns=[
                {"incomplete_vars": [3], "weights": [0, 4, 1, 0]},
                {"incomplete_vars": [2]},
                {"incomplete_vars": [1, 2], "mechanism": "MNAR"},
                {
                    "incomplete_vars": [1, 2, 3],
                    "weights": {0: -2, 3: 1},
                    "mechanism": "MAR+MNAR",
                },
            ]
        )

        nhanes2_incomplete = ma.fit_transform(self.nhanes2_sim)

        mdp = mdPatterns()
        mypatterns = mdp.get_patterns(nhanes2_incomplete, show_plot=False)

        self.assertListEqual(
            ma.weights.tolist(),
            [[0, 4, 1, 0], [1, 1, 0, 1], [0, 1, 1, 0], [-2, 0, 0, 1]],
        )

        self.assertTrue(len(ma.wss_per_pattern), 4)


if __name__ == "__main__":
    unittest.main()
