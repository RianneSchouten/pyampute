import numpy as np
import unittest

from pyampute.ampute import MultivariateAmputation

# test that all mechanisms work
class TestAmpute(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_mechanisms(self):

        # create complete data
        n = 1000
        X = np.random.randn(n, 2)

        for mechanism in ["MAR", "MNAR", "MCAR"]:
            current_mechanisms = np.repeat(mechanism, 2)
            ma = MultivariateAmputation(
                patterns=[
                    {"incomplete_vars": [i], "mechanism": mechanism}
                    for i, mechanism in enumerate(current_mechanisms)
                ]
            )
            incomplete_data = ma.fit_transform(X)
            self.assertEqual(incomplete_data.shape, X.shape)

            count_missing_values_per_column = np.sum(np.isnan(incomplete_data), axis=0)
            self.assertTrue(
                np.all(count_missing_values_per_column > (0.4 * 0.5 * n))
            )  # expect: around 250
            self.assertGreater(
                np.sum(count_missing_values_per_column), (0.4 * n)
            )  # expect: around 500

            # check if it also works if len(mechanisms) = 1
            ma = MultivariateAmputation(
                patterns=[{"incomplete_vars": [0], "mechanism": mechanism}]
            )
            incomplete_data = ma.fit_transform(X)
            self.assertTrue(
                np.all(count_missing_values_per_column > (0.4 * 0.5 * n))
            )  # expect: around 250
            self.assertGreater(
                np.sum(count_missing_values_per_column), (0.4 * n)
            )  # expect: around 500

    # test one specific situation
    def test_specific_situation(self):
        # create complete data
        n = 10000
        X = np.random.randn(n, 2)

        # define some arguments
        my_incomplete_vars = [np.array([0]), np.array([1]), np.array([1])]
        my_freqs = np.array((0.3, 0.2, 0.5))
        my_weights = [np.array([4, 1]), np.array([0, 1]), np.array([1, 0])]
        my_prop = 0.3

        patterns = [
            {"incomplete_vars": incomplete_vars, "freq": freq, "weights": weights}
            for incomplete_vars, freq, weights in zip(
                my_incomplete_vars, my_freqs, my_weights
            )
        ]

        # run ampute
        ma = MultivariateAmputation(prop=my_prop, patterns=patterns)
        incomplete_data = ma.fit_transform(X)
        print(incomplete_data)
        self.assertEqual(incomplete_data.shape, X.shape)

        # print(np.sum(np.sum(np.isnan(incomplete_data), axis=0))) # expect: around 3000
        # print(np.sum(np.isnan(incomplete_data), axis=0)[0]) # expect: around 2100
        # print(np.sum(np.isnan(incomplete_data), axis=0)[1]) # expect: around 900

        self.assertLess(
            np.absolute(
                (my_prop * len(X)) - np.sum(np.sum(np.isnan(incomplete_data), axis=0))
            ),
            100,
        )
        self.assertLess(
            np.absolute(
                (0.3 * my_prop * len(X)) - np.sum(np.isnan(incomplete_data), axis=0)[0]
            ),
            100,
        )
        self.assertLess(
            np.absolute(
                (0.7 * my_prop * len(X)) - np.sum(np.isnan(incomplete_data), axis=0)[1]
            ),
            100,
        )


if __name__ == "__main__":
    unittest.main()
