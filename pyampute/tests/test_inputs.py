import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch

# Local imports
from pyampute.ampute import MultivariateAmputation
from pyampute.exploration.md_patterns import mdPatterns

seed = 0
columns = ["age", "weight", "ismale", "fries_s", "fries_m", "fries_l"]
X_nomissing = pd.DataFrame(
    [
        [44, 15.1, 0, 0, 1, 0],
        [49, 57.2, 1, 0, 0, 1],
        [26, 26.3, 0, 0, 1, 0],
        [16, 73.4, 1, 1, 0, 0],
        [13, 56.5, 1, 0, 1, 0],
        [57, 29.6, 0, 1, 0, 0],
    ],
    columns=columns,
)
continuous_columns = ["age", "weight"]
onehot_prefix_names = ["fries"]
y = pd.Series([1, 0, 1, 0, 1, 0])

standard = {
    "X": X_nomissing,
    "y": y,
    "seed": seed,
    "val_test_size": 0.5,
    "test_size": 0.5,
}


class TestDefaults(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_minimal_defaults(self):
        minimal = MultivariateAmputation(seed=4)
        minimal._validate_input(X_nomissing)
        self.assertTrue(np.array_equal(minimal.freqs, np.array([1]),))
        self.assertTrue(np.array_equal(minimal.mechanisms, np.array(["MAR"])))
        self.assertTrue(
            np.array_equal(
                minimal.score_to_probability_func, np.array(["SIGMOID-RIGHT"])
            )
        )
        X_amputed = minimal.fit_transform(X_nomissing)
        mdp = mdPatterns()
        patterns = mdp.get_patterns(X_amputed, show_plot=False)
        # There should be approximately half the rows missing data (via prop=0.5)
        self.assertTrue(
            patterns[1, "row_count"] == X_nomissing.shape[0] // 2
            or patterns.loc[1, "row_count"] == (X_nomissing.shape[0] // 2 + 1)
        )
        # half of the vars should have missing values
        self.assertTrue(
            patterns.loc[1, "n_missing_values"] / X_nomissing.shape[1] == 3 / 6
        )

    def test_adjusting_inputs(self):
        with self.subTest("Adjust Primitive Defaults"):
            # test no passing any freq
            patterns = [
                # Test lowercase names, named indices, no weights
                {
                    "incomplete_vars": ["age"],
                    "mechanism": "mar",
                    "score_to_probability_func": "sigmoid-left",
                },
                # test mixed case names, integer indices, custom weight, mar+mnar
                {
                    "incomplete_vars": [0, 3],
                    "mechanism": "maR+mNar",
                    "score_to_probability_func": "sigmoid-mid",
                    "weights": [0.5, 1, 0, 0, 0, 0],
                },
            ]
            adjust = MultivariateAmputation(prop=45, patterns=patterns)
            adjust._validate_input(X_nomissing)

            self.assertEqual(adjust.prop, 0.45)
            self.assertTrue(
                np.array_equal(
                    adjust.observed_var_indicator,
                    np.array([[0, 1, 1, 1, 1, 1], [0, 1, 1, 0, 1, 1]]),
                )
            )
            self.assertTrue(
                np.array_equal(adjust.mechanisms, np.array(["MAR", "MAR+MNAR"]),)
            )
            self.assertTrue(
                np.array_equal(
                    adjust.score_to_probability_func,
                    np.array(["SIGMOID-LEFT", "SIGMOID-MID"]),
                )
            )

    def test_optional_args(self):
        # includes testing no freqs passed
        patterns = [
            {"incomplete_vars": [0]},
            {"incomplete_vars": [1], "mechanism": "MCAR"},
            {"incomplete_vars": [2], "score_to_probability_func": "sigmoid-mid"},
        ]
        mechanism_case_coverage = MultivariateAmputation(patterns=patterns)
        mechanism_case_coverage._validate_input(X_nomissing)
        self.assertTrue(
            np.array_equal(
                mechanism_case_coverage.observed_var_indicator,
                np.array([[0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1]]),
            )
        )
        self.assertTrue(
            np.array_equal(
                mechanism_case_coverage.mechanisms, np.array(["MAR", "MCAR", "MAR"]),
            )
        )
        self.assertTrue(
            np.array_equal(
                mechanism_case_coverage.score_to_probability_func,
                np.array(["SIGMOID-RIGHT", "SIGMOID-RIGHT", "SIGMOID-MID"]),
            )
        )
        self.assertTrue(
            np.array_equal(mechanism_case_coverage.freqs, np.array([1 / 3] * 3),)
        )

    def test_weights_dict(self):
        # test names
        patterns = [
            {
                "incomplete_vars": [0],
                "mechanism": "mar+mnar",
                "weights": {name: 1 for name in columns[:-2]},
            }
        ]
        mechanism_case_coverage = MultivariateAmputation(patterns=patterns)
        mechanism_case_coverage._validate_input(X_nomissing)
        self.assertTrue(
            np.array_equal(
                mechanism_case_coverage.weights, np.array([[1, 1, 1, 1, 0, 0]]),
            )
        )

        # test indices
        patterns = [
            {
                "incomplete_vars": [0],
                "mechanism": "mar+mnar",
                "weights": {i: 1 for i in [0, 2, 4]},
            }
        ]
        mechanism_case_coverage = MultivariateAmputation(patterns=patterns)
        mechanism_case_coverage._validate_input(X_nomissing)
        self.assertTrue(
            np.array_equal(
                mechanism_case_coverage.weights, np.array([[1, 0, 1, 0, 1, 0]]),
            )
        )

    def test_mechanism_case_coverage(self):
        # TODO: test for repeat patterns?
        mar_mnar_weights = [0, 0, 0, 0.5, 1, 0]
        patterns = [
            {"incomplete_vars": [0], "mechanism": "mcar"},
            {"incomplete_vars": [1], "mechanism": "mar"},
            {"incomplete_vars": [2], "mechanism": "mnar"},
            {
                "incomplete_vars": [3],
                "mechanism": "mar+mnar",
                "weights": mar_mnar_weights,
            },
        ]
        mechanism_case_coverage = MultivariateAmputation(patterns=patterns)
        mechanism_case_coverage._validate_input(X_nomissing)

        self.assertTrue(
            np.array_equal(
                mechanism_case_coverage.weights,
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 1, 1],
                        [0, 0, 1, 0, 0, 0],
                        mar_mnar_weights,
                    ]
                ),
            )
        )


class TestBadArgs(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_bad_data(self):
        amputer = MultivariateAmputation()
        # data can't be empty
        with self.assertRaises(AssertionError):
            amputer._validate_input(None)
        # must be 2d
        with self.assertRaises(AssertionError):
            amputer._validate_input(X_nomissing.iloc[0])
        # more than one column required
        with self.assertRaises(AssertionError):
            amputer._validate_input(X_nomissing.iloc[0, :])
        # data cannot have missing values for vars involved in ampute
        with self.assertRaises(AssertionError):
            X = X_nomissing.copy()
            # first column (first value) missing value
            X.iloc[0, 0] = np.nan
            # first column involved in amputation, by default when column 1 is missing column 0 will be assigned a weight of 1
            amputer = MultivariateAmputation(patterns=[{"incomplete_vars": [1]}])
            amputer._validate_input(X)

    def test_bad_incomplete_vars(self):
        bad_patterns = [
            # no features to be amputed
            [{"incomplete_vars": []}],
            # num of vars in patterns too many
            [{"incomplete_vars": list(range(15))}],
            # bad indices (int)
            [{"incomplete_vars": [0, 15]}],
            # bad indices (name)
            [{"incomplete_vars": ["age", "burger"]}],
            # all vars missing does nothing under MAR
            [
                {
                    "incomplete_vars": np.array(range(X_nomissing.shape[1])),
                    "mechanism": "MAR",
                }
            ],
        ]

        for patterns in bad_patterns:
            with self.assertRaises(AssertionError):
                MultivariateAmputation(patterns=patterns)._validate_input(X_nomissing)

    def test_bad_prop(self):
        # 0 and 100 fine
        with self.assertRaises(AssertionError):
            MultivariateAmputation(prop=-3.2)._validate_input(X_nomissing)
        with self.assertRaises(AssertionError):
            MultivariateAmputation(prop=324)._validate_input(X_nomissing)

    def test_bad_freqs(self):
        bad_patterns = [
            # can't be all 0s, needs to sum to 1
            [{"incomplete_vars": [0], "freq": 0}],
            [{"incomplete_vars": [0], "freq": 2}],
            # must be between 0 and 1 (even though it sums to 1)
            [
                {"incomplete_vars": [ivs], "freq": f}
                for ivs, f in zip(range(6), [0.2, 3, 0.1, -3, 0.4, 0.3])
            ],
            # must sum to 1
            [
                {"incomplete_vars": [ivs], "freq": f}
                for ivs, f in zip(range(6), [0.2, 0.3, 0.1, 0.3, 0.4, 0.3])
            ],
            # cannot specify only some
            [{"incomplete_vars": [0], "freq": 0}, {"incomplete_vars": [1]}],
        ]
        for patterns in bad_patterns:
            with self.assertRaises(AssertionError):
                MultivariateAmputation(patterns=patterns)._validate_input(X_nomissing)

    def test_bad_mechanisms(self):
        # invalid names
        with self.assertRaises(AssertionError):
            MultivariateAmputation(
                patterns=[{"incomplete_vars": [0], "mechanism": "MARP"}]
            )._validate_input(X_nomissing)

    def test_bad_score_to_probabiliyt_func(self):
        # bad name
        with self.assertRaises(AssertionError):
            MultivariateAmputation(
                patterns=[
                    {
                        "incomplete_vars": [0],
                        "score_to_probability_func": "smigmoid-up",
                    }
                ]
            )._validate_input(X_nomissing)

    def test_bad_weights(self):
        bad_patterns = [
            # needs to be a list,
            {"incomplete_vars": [0]},
            # shape must match num vars
            [{"incomplete_vars": [0], "weights": list(range(2))}],
            [{"incomplete_vars": [0], "weights": list(range(15))}],
            # MCAR should have weights all 0s
            [
                {
                    "incomplete_vars": [0],
                    "mechanism": "MCAR",
                    "weights": [0, 1, 1, 1, 1, 1],
                }
            ],
            # cannot get default weights for mar+mnar
            [{"incomplete_vars": [0], "mechanism": "mar+mnar"}],
            # dict form: too many weights
            [
                {
                    "incomplete_vars": [0],
                    "mechanism": "mar+mnar",
                    "weights": {i: 1 for i in range(15)},
                }
            ],
            # dict form: bad indices
            [{"incomplete_vars": [0], "mechanism": "mar+mnar", "weights": {15: 1}}],
            # dict form: bad name
            [
                {
                    "incomplete_vars": [0],
                    "mechanism": "mar+mnar",
                    "weights": {"burger": 1},
                }
            ],
        ]

        for patterns in bad_patterns:
            with self.assertRaises(AssertionError):
                MultivariateAmputation(patterns=patterns)._validate_input(X_nomissing)


if __name__ == "__main__":
    unittest.main()
