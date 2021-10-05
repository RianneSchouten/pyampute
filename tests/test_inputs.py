import pandas as pd
import numpy as np
import unittest

"""
https://docs.python.org/3/library/unittest.mock.html#where-to-patch
patch where an object is looked up, not where it is defined
from unittest.mock import patch
"""

from pymice.amputation.ampute import MultivariateAmputation

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

default_missing_pattern = np.array(
    [
        [0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 0],
    ]
)


class TestDefaults(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_minimal_defaults(self):
        minimal = MultivariateAmputation()
        minimal._validate_input(X_nomissing)
        a_sixth = 1 / 6
        self.assertTrue(np.array_equal(minimal.patterns, default_missing_pattern))
        self.assertTrue(
            np.array_equal(
                minimal.freqs,
                np.array([a_sixth, a_sixth, a_sixth, a_sixth, a_sixth, a_sixth]),
            )
        )
        self.assertTrue(
            np.array_equal(
                minimal.mechanisms, np.array(["MAR", "MAR", "MAR", "MAR", "MAR", "MAR"])
            )
        )
        self.assertTrue(
            np.array_equal(
                minimal.types,
                np.array(["RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"]),
            )
        )
        self.assertTrue(np.array_equal(minimal.weights, default_missing_pattern))

    def test_adjusting_inputs(self):
        with self.subTest("Adjust Primitive Defaults"):
            adjust = MultivariateAmputation(prop=45, mechanisms="mar", types="right")
            adjust._validate_input(X_nomissing)

            self.assertEqual(adjust.prop, 0.45)
            self.assertTrue(
                np.array_equal(
                    adjust.mechanisms,
                    np.array(["MAR", "MAR", "MAR", "MAR", "MAR", "MAR"]),
                )
            )
            self.assertTrue(
                np.array_equal(
                    adjust.types,
                    np.array(["RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"]),
                )
            )

            # mixed case fine too
            adjust_mixed = MultivariateAmputation(mechanisms="mAr", types="rIGht")
            adjust_mixed._validate_input(X_nomissing)
            self.assertTrue(
                np.array_equal(
                    adjust_mixed.mechanisms,
                    np.array(["MAR", "MAR", "MAR", "MAR", "MAR", "MAR"]),
                )
            )
            self.assertTrue(
                np.array_equal(
                    adjust_mixed.types,
                    np.array(["RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"]),
                )
            )

        with self.subTest("Adjust Broadcasting Primitives"):
            broadcast_primitives = MultivariateAmputation(
                freqs=1 / 6, mechanisms="MCAR", types="TAIL"
            )
            # default 6 patterns
            broadcast_primitives._validate_input(X_nomissing)

            self.assertTrue(
                np.array_equal(
                    broadcast_primitives.freqs,
                    np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]),
                )
            )
            self.assertTrue(
                np.array_equal(
                    broadcast_primitives.mechanisms,
                    np.array(["MCAR", "MCAR", "MCAR", "MCAR", "MCAR", "MCAR"]),
                )
            )
            self.assertTrue(
                np.array_equal(
                    broadcast_primitives.types,
                    np.array(["TAIL", "TAIL", "TAIL", "TAIL", "TAIL", "TAIL"]),
                )
            )

        with self.subTest("Adjust Broadcasting 1 List Item"):
            adjust_broadcast = MultivariateAmputation(
                freqs=[1 / 6], mechanisms=["MCAR"], types=["TAIL"]
            )
            # default 6 patterns
            adjust_broadcast._validate_input(X_nomissing)

            self.assertTrue(
                np.array_equal(
                    adjust_broadcast.freqs,
                    np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]),
                )
            )
            self.assertTrue(
                np.array_equal(
                    adjust_broadcast.mechanisms,
                    np.array(["MCAR", "MCAR", "MCAR", "MCAR", "MCAR", "MCAR"]),
                )
            )
            self.assertTrue(
                np.array_equal(
                    adjust_broadcast.types,
                    np.array(["TAIL", "TAIL", "TAIL", "TAIL", "TAIL", "TAIL"]),
                )
            )

    def test_mechanism_case_coverage(self):
        mechanism_case_coverage = MultivariateAmputation(
            mechanisms=["MCAR", "MAR", "MNAR", "MCAR", "MAR", "MNAR"]
        )
        mechanism_case_coverage._validate_input(X_nomissing)

        self.assertTrue(
            np.array_equal(
                mechanism_case_coverage.weights,
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 1, 1],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 1],
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

    def test_bad_patterns(self):
        # bad shape
        with self.assertRaises(AssertionError):
            MultivariateAmputation(patterns=np.array([0, 1, 1]))._validate_input(
                X_nomissing
            )
        # num of vars in patterns don't match the num of vars in data
        with self.assertRaises(AssertionError):
            MultivariateAmputation(
                patterns=np.array([[0, 1, 1], [0, 0, 1]])
            )._validate_input(X_nomissing)
        # only 0s/1s
        with self.assertRaises(AssertionError):
            MultivariateAmputation(
                patterns=np.array([0, 3, 1, 0, 1, 1])
            )._validate_input(X_nomissing)
        # all 1s does nothing
        with self.assertRaises(AssertionError):
            MultivariateAmputation(
                patterns=np.array([1, 1, 1, 1, 1, 1])
            )._validate_input(X_nomissing)
        # MAR needs at least one observed var
        with self.assertRaises(AssertionError):
            MultivariateAmputation(
                mechanisms="MAR", patterns=np.array([0, 0, 0, 0, 0, 0])
            )._validate_input(X_nomissing)

    def test_bad_prop(self):
        # 0 and 100 fine
        with self.assertRaises(AssertionError):
            MultivariateAmputation(prop=-3.2)._validate_input(X_nomissing)
        with self.assertRaises(AssertionError):
            MultivariateAmputation(prop=324)._validate_input(X_nomissing)

    def test_bad_freqs(self):
        # can't be all 0s or 1s, needs to sum to 1
        with self.assertRaises(AssertionError):
            MultivariateAmputation(freqs=1)._validate_input(X_nomissing)
        with self.assertRaises(AssertionError):
            MultivariateAmputation(freqs=0)._validate_input(X_nomissing)
        with self.assertRaises(AssertionError):
            MultivariateAmputation(freqs=2)._validate_input(X_nomissing)

        # must be between 0 and 1 (even though it sums to 1)
        with self.assertRaises(AssertionError):
            MultivariateAmputation(freqs=[0.2, 3, 0.1, -3, 0.4, 0.3])._validate_input(
                X_nomissing
            )
        # must sum to 1
        with self.assertRaises(AssertionError):
            MultivariateAmputation(
                freqs=[0.2, 0.3, 0.1, 0.3, 0.4, 0.3]
            )._validate_input(X_nomissing)

    def test_bad_mechanisms(self):
        # default is 6 patterns for X_nomissing, must have the same # mechanisms
        with self.assertRaises(AssertionError):
            MultivariateAmputation(mechanisms=["MCAR", "MAR", "MNAR"])._validate_input(
                X_nomissing
            )
        # invalid names
        with self.assertRaises(AssertionError):
            MultivariateAmputation(
                mechanisms=["MCAR", "MARP", "MNAR", "MAR", "MCAR", "MAR"]
            )._validate_input(X_nomissing)
        with self.assertRaises(AssertionError):
            MultivariateAmputation(mechanisms="MARP")._validate_input(X_nomissing)

    def test_bad_types(self):
        # default is 6 patterns for X_nomissing, 6 mechanisms, must have same # types
        with self.assertRaises(AssertionError):
            MultivariateAmputation(types=["right", "left"])._validate_input(X_nomissing)
        # invalid names
        with self.assertRaises(AssertionError):
            MultivariateAmputation(
                mechanisms=["right", "sright", "left", "mid", "tail", "mid"]
            )._validate_input(X_nomissing)
        with self.assertRaises(AssertionError):
            MultivariateAmputation(mechanisms="sright")._validate_input(X_nomissing)

    def test_bad_weights(self):
        # shape must match patterns
        with self.assertRaises(AssertionError):
            MultivariateAmputation(
                weights=default_missing_pattern[:5, :]
            )._validate_input(X_nomissing)
        with self.assertRaises(AssertionError):
            MultivariateAmputation(
                weights=default_missing_pattern[:, :5]
            )._validate_input(X_nomissing)
        with self.assertRaises(AssertionError):
            MultivariateAmputation(
                weights=default_missing_pattern[:5, :5]
            )._validate_input(X_nomissing)

        # MCAR should have weights all 0s
        with self.assertRaises(AssertionError):
            MultivariateAmputation(
                mechanisms="MCAR", weights=default_missing_pattern
            )._validate_input(X_nomissing)


if __name__ == "__main__":
    unittest.main()
