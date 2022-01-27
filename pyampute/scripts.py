import numpy as np
import pandas as pd


from pyampute.ampute import MultivariateAmputation
from pyampute.utils import LOOKUP_TABLE_PATH


def generate_shift_lookup_table(
    lookup_table_path: str = LOOKUP_TABLE_PATH,
    n_samples: int = int(1e6),
    lower_range: float = MultivariateAmputation.DEFAULTS["lower_range"],
    upper_range: float = MultivariateAmputation.DEFAULTS["upper_range"],
    max_iter: int = MultivariateAmputation.DEFAULTS["max_iter"],
    max_diff_with_target: float = MultivariateAmputation.DEFAULTS[
        "max_diff_with_target"
    ],
):
    normal_sample = np.random.standard_normal(size=n_samples)
    percent_missing = np.arange(0.01, 1.01, 0.01)
    score_to_prob_func_names = [
        "SIGMOID-RIGHT",
        "SIGMOID-LEFT",
        "SIGMOID-TAIL",
        "SIGMOID-MID",
    ]
    shifts = []
    for func in score_to_prob_func_names:
        shifts.append(
            [
                MultivariateAmputation._binary_search(
                    normal_sample,
                    func,
                    percent,
                    lower_range,
                    upper_range,
                    max_iter,
                    max_diff_with_target,
                )[0]
                for percent in percent_missing
            ]
        )
    lookup_table = pd.DataFrame(
        shifts, index=score_to_prob_func_names, columns=percent_missing,
    )
    lookup_table.to_csv(lookup_table_path)


if __name__ == "__main__":
    generate_shift_lookup_table()
