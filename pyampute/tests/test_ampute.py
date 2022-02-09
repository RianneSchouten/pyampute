import numpy as np
import unittest

from pyampute.ampute import MultivariateAmputation
from pyampute.exploration.md_patterns import mdPatterns


# test that all mechanisms work
class TestAmpute(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_mechanisms(self):
        # create complete data
        n = 10000
        X = np.random.randn(n, 2)

        for mechanism in ["MAR", "MNAR", "MCAR"]:
            with self.subTest("Multiple Patterns"):
                # Testing > 1 pattern
                current_mechanisms = np.repeat(mechanism, 2)
                ma = MultivariateAmputation(
                    patterns=[
                        {"incomplete_vars": [i], "mechanism": mechanism}
                        for i, mechanism in enumerate(current_mechanisms)
                    ]
                )
                X_amputed = ma.fit_transform(X)
                self.assertEqual(X_amputed.shape, X.shape)

                mdp = mdPatterns()
                patterns = mdp.get_patterns(X_amputed, show_plot=False)

                self.assertTrue(
                    # each column should have prop*freq missing, 2 patterns so freq for each is 50%
                    np.allclose(
                        patterns["row_count"].iloc[1:-1].astype(int),
                        (0.5 * 0.5 * n),
                        atol=0.05 * n,
                    )
                )  # expect: around 250
                # about half the rows should be missing values
                self.assertAlmostEqual(
                    patterns.loc["n_missing_values_per_col", "n_missing_values"].astype(
                        int
                    ),
                    0.5 * n,
                    delta=0.05 * n,
                )  # expect: around 500

            # check if it also works if len(mechanisms) = 1
            with self.subTest("1 Mechanism / 1 Pattern"):
                ma = MultivariateAmputation(
                    patterns=[{"incomplete_vars": [0], "mechanism": mechanism}]
                )
                X_amputed = ma.fit_transform(X)
                mdp = mdPatterns()
                patterns = mdp.get_patterns(X_amputed, show_plot=False)
                self.assertAlmostEqual(
                    # column 0 should have prop% missing
                    patterns.loc["n_missing_values_per_col", 0],
                    0.5 * n,
                    delta=0.05 * n,
                )
                #  column 1 should have none missing
                self.assertEqual(patterns.loc["n_missing_values_per_col", 1], 0)
                # about half the rows should be missing values
                self.assertAlmostEqual(
                    patterns.loc["n_missing_values_per_col", "n_missing_values"].astype(
                        int
                    ),
                    0.5 * n,
                    delta=0.05 * n,
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
        X_amputed = ma.fit_transform(X)
        self.assertEqual(X_amputed.shape, X.shape)

        # print(np.sum(np.sum(np.isnan(incomplete_data), axis=0))) # expect: around 3000
        # print(np.sum(np.isnan(incomplete_data), axis=0)[0]) # expect: around 2100
        # print(np.sum(np.isnan(incomplete_data), axis=0)[1]) # expect: around 900
        mdp = mdPatterns()
        patterns = mdp.get_patterns(X_amputed, show_plot=False)
        # about 30% rows should be missing values
        self.assertAlmostEqual(
            patterns.loc["n_missing_values_per_col", "n_missing_values"].astype(int),
            0.3 * n,
            delta=0.05 * n,
        )
        # both columns should be missing values
        self.assertTrue(
            np.allclose(
                patterns.loc["n_missing_values_per_col"].iloc[1:-1].astype(int),
                [(0.3 * 0.3 * n), (0.3 * (0.2 + 0.5) * n)],
                atol=0.05 * n,
            )
        )

    def test_repeat_pattern(self):
        n = 10000
        with self.subTest("1 Repeat Pattern on Same Var"):
            X = np.random.randn(n, 2)
            patterns = [
                {"incomplete_vars": [0], "mechanism": "mcar"},
                {"incomplete_vars": [0], "mechanism": "mcar"},
            ]
            repeat_patterns = MultivariateAmputation(patterns=patterns, prop=0.3)
            X_amputed = repeat_patterns.fit_transform(X)
            mdp = mdPatterns()
            patterns = mdp.get_patterns(X_amputed, show_plot=False)

            # total number of incomplete rows equals prop
            self.assertTrue(
                np.allclose(
                    patterns.loc[1, "row_count"], (0.3 * n), atol=0.05 * n,
                )
            )
        with self.subTest(
            "Repeat Pattern on Same Var, Varying Freq Plus Extra Pattern"
        ):
            X = np.random.randn(n, 2)
            new_patterns = [
                {"incomplete_vars": [0], "mechanism": "mcar", "freq": 0.2},
                {"incomplete_vars": [0], "mechanism": "mcar", "freq": 0.1},
                {"incomplete_vars": [1], "mechanism": "mar", "freq": 0.7},
            ]
            repeat_patterns = MultivariateAmputation(patterns=new_patterns, prop=0.6)
            X_amputed = repeat_patterns.fit_transform(X)
            mdp = mdPatterns()
            patterns = mdp.get_patterns(X_amputed, show_plot=False)

            # total number of incomplete rows equals prop
            self.assertTrue(
                np.allclose(
                    patterns.loc[2, "row_count"], ((0.2 + 0.1) * n * 0.6), atol=0.05 * n,
                )
            )

    def test_seed(self):
        # create complete data
        n = 10000
        X = np.random.randn(n, 2)
        default = MultivariateAmputation()  # no seed set by default
        # should produce different values
        self.assertFalse(
            np.array_equal(
                default.fit_transform(X), default.fit_transform(X), equal_nan=True
            )
        )

        default = MultivariateAmputation(seed=4)  # seed set
        # should produce same values
        self.assertTrue(
            np.array_equal(
                default.fit_transform(X), default.fit_transform(X), equal_nan=True
            )
        )

    def test_self(self):

        n = 10000
        X = np.random.randn(n, 10)
    
        ma = MultivariateAmputation(
            patterns = [
                {'incomplete_vars': [2,3], 'mechanism': "MCAR"},
                {'incomplete_vars': [8,9,1], 'mechanism': "MNAR"},
                {'incomplete_vars': [1], 'weights': {0:1,1:2}, 'mechanism': "MAR+MNAR"}
            ]
        )
        X_amputed = ma.fit_transform(X)

        self.assertTrue(len(ma.wss_per_pattern), 3)
        self.assertTrue(len(ma.probs_per_pattern), 3)
        self.assertListEqual(ma.mechanisms.tolist(), ["MCAR", "MNAR", "MAR+MNAR"])
        self.assertEqual(len(ma.wss_per_pattern[0]), len(ma.probs_per_pattern[0]))
        self.assertEqual(len(ma.assigned_group_number), X.shape[0])
        self.assertEqual(len(ma.assigned_group_number[ma.assigned_group_number == 1]), len(ma.wss_per_pattern[1]))
        self.assertEqual(len(ma.assigned_group_number[ma.assigned_group_number == 2]), len(ma.probs_per_pattern[2]))

    def test_sigmoid_score_to_prob_function(self):
        # create complete data
        n = 10000
        X = np.random.randn(n, 2)

        my_score_to_prob_functions = [
            "sigmoid-right",
            "sigmoid-left",
            "sigmoid-mid",
            "sigmoid-tail",
        ]
        my_prop = 0.3

        for score_to_prob_function in my_score_to_prob_functions:
            patterns = [
                {
                    "incomplete_vars": np.array([0]),
                    "score_to_probability_func": score_to_prob_function,
                }
            ]

            # run ampute
            ma = MultivariateAmputation(prop=my_prop, patterns=patterns)
            X_amputed = ma.fit_transform(X)

            mdp = mdPatterns()
            patterns = mdp.get_patterns(X_amputed, show_plot=False)
            # about 30% rows should be missing values
            self.assertAlmostEqual(
                patterns.loc["n_missing_values_per_col", "n_missing_values"].astype(
                    int
                ),
                0.3 * n,
                delta=0.05 * n,
            )
            # first column only should be missing values
            self.assertTrue(
                np.allclose(
                    patterns.loc["n_missing_values_per_col"]
                    .iloc[1:-1]
                    .astype(int)
                    .sort_index(),  # For some reason the patterns can appear out of order
                    [(0.3 * n), 0],
                    atol=0.05 * n,
                )
            )
            probs = ma.probs_per_pattern[0]
            wss = ma.wss_per_pattern[0]
            if score_to_prob_function == "sigmoid-right":
                self.assertGreater(
                    probs[np.argmax(wss)], probs[np.argmin(wss)],
                )
            elif score_to_prob_function == "sigmoid-left":
                self.assertGreater(
                    probs[np.argmin(wss)], probs[np.argmax(wss)],
                )
            elif score_to_prob_function == "sigmoid-mid":
                argmedian = np.argsort(wss)[len(wss) // 2]
                self.assertGreater(
                    probs[argmedian], probs[np.argmax(wss)],
                )
                self.assertGreater(
                    probs[argmedian], probs[np.argmin(wss)],
                )
            else:  # tail
                argmedian = np.argsort(wss)[len(wss) // 2]
                self.assertGreater(
                    probs[np.argmax(wss)], probs[argmedian],
                )
                self.assertGreater(
                    probs[np.argmin(wss)], probs[argmedian],
                )


if __name__ == "__main__":
    unittest.main()
