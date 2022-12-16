"""Displays missing data patterns in incomplete datasets"""
# Author: Rianne Schouten <https://rianneschouten.github.io/>
# Co-Author: Srinidhi Ilango

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

# Local
from pyampute.utils import Matrix


class mdPatterns:
    """
    Displays missing data patterns in incomplete datasets

    Extracts all unique missing data patterns in an incomplete dataset and creates a visualization. ``1`` (red) and ``0`` (blue) refer to missing and observed values respectively.

    Parameters
    ----------
    None: currently no parameters available.

    Attributes
    ----------
    md_patterns : pandas DataFrame of shape `(k+2, m+2)`
        `k` is the number of unique missing data patterns and `m` the number of dataset columns (features). ``0`` and ``1`` correspond to missing and observed values respectively. The first row displays the data rows with no missing values and the last row gives column totals. The first column displays the count or proportion of rows that follow a pattern and the last column displays the number of missing values per pattern.

    See also
    --------
    :class:`~pyampute.ampute.MultivariateAmputation` : Transformer for generating multivariate missingness in complete datasets

    Notes
    -----
    This class is useful for investigating any structure in an incomplete dataset, and can help to understand possible reasons or solutions. We follow the logic of a comparable R-function, `mice::md_patterns`_.

    .. _`mice::md_patterns`: https://github.com/amices/mice

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pyampute.exploration.md_patterns import mdPatterns
    >>> nhanes2 = pd.read_csv("data/nhanes2.csv")
    >>> mdp = mdPatterns()
    >>> patterns = mdp.get_patterns(nhanes2)
    >>> print(patterns)
                            row_count  age  hyp  bmi  chl  n_missing_values
    rows_no_missing                 13    1    1    1    1                 0
    1                                3    1    1    1    0                 1
    2                                1    1    1    0    1                 1
    3                                1    1    0    0    1                 2
    4                                7    1    0    0    0                 3
    n_missing_values_per_col              0    8    9   10                27

    """

    def __init__(self):

        # initialize empty attributes
        self.md_patterns = None

    def get_patterns(
        self, X: Matrix, count_or_proportion: str = "count", show_plot: bool = True
    ) -> pd.DataFrame:
        """Extracts and visualizes missing data patterns in an incomplete dataset

        Parameters
        ----------
        X : Matrix of shape `(n, m)`
            Dataset with missing values. `n` rows (samples) and `m` columns (features).
        
        count_or_proportion : str, {"count", "proportion"}, default : "count"
            Whether the number of rows should be specified as a count or a proportion. 

        show_plot : bool, default : True
            Whether a plot should be displayed using ``plt.show``. 

        Returns
        -------
        md_patterns : pandas DataFrame of shape `(k+2, m+2)`
            `k` is the number of unique missing data patterns and `m` the number of dataset columns (features). 
            The first row displays the data rows with no missing values and the last row gives column totals. 
            The first column displays the count or proportion of rows that follow a pattern, 
            the last column displays the number of missing values per pattern.
        """

        # make sure X is a pd.DataFrame
        Xdf = pd.DataFrame(X)

        # calculate patterns
        self._calculate_patterns(Xdf, count_or_proportion)

        # make plot
        if show_plot:
            self._make_plot()

        return self.md_patterns

    def _calculate_patterns(
        self, X: pd.DataFrame, count_or_proportion: str = "count"
    ) -> pd.DataFrame:
        """Extracts all unique missing data patterns in an incomplete dataset and transforms into a pandas DataFrame"""

        # mask
        mask = X.isnull()

        # count number of missing values per column
        colsums = mask.sum()
        sorted_col = colsums.sort_values().index.tolist()
        colsums["n_missing_values"] = colsums.sum()
        colsums["row_count"] = ""

        # finding missing values per group
        group_values = (~mask).groupby(sorted_col).size().reset_index(name="row_count")
        group_values["n_missing_values"] = group_values.isin([0]).sum(axis=1)
        group_values.sort_values(
            by=["n_missing_values", "row_count"], ascending=[True, False], inplace=True
        )
        group_values = group_values.append(colsums, ignore_index=True)

        # add extra row to patterns when there are no incomplete rows in dataset
        if group_values.iloc[0, 0:-2].values.tolist() != list(np.ones(len(sorted_col))):
            group_values.loc[-1] = np.concatenate(
                (np.ones(len(sorted_col)), np.zeros(2))
            ).astype(int)
            group_values.index = group_values.index + 1  # shifting index
            group_values.sort_index(inplace=True)

            # put row_count in the begining
        cols = list(group_values)
        cols.insert(0, cols.pop(cols.index("row_count")))
        group_values = group_values.loc[:, cols]

        if count_or_proportion == "proportion":
            group_values.rename(columns={"row_count": "row_prop"}, inplace=True)
            percents = ((group_values.iloc[0:-1, 0]).astype(int) / X.shape[0]).round(2)
            group_values.iloc[0:-1, 0] = percents.astype(str)
            group_values.iloc[-1, 1:-1] = group_values.iloc[-1, 1:-1] / X.shape[0]
            group_values.iloc[-1, -1] = (
                group_values.iloc[-1, -1] / (X.shape[0] * X.shape[1])
            ).round(2)

        self.md_patterns = group_values
        self.md_patterns.index = (
            ["rows_no_missing"]
            + list(self.md_patterns.index[1:-1])
            + ["n_missing_values_per_col"]
        )
        return self.md_patterns

    def _make_plot(self):
        """"Creates visualization of missing data patterns"""

        group_values = self.md_patterns

        heat_values = group_values.iloc[
            0 : (group_values.shape[0] - 1), 1 : group_values.shape[1] - 1
        ]

        myred = "#B61A51B3"
        myblue = "#006CC2B3"
        cmap = colors.ListedColormap([myred, myblue])

        fig, ax = plt.subplots(1)
        ax.imshow(heat_values.astype(bool), aspect="auto", cmap=cmap)

        by = ax.twinx()  # right ax
        bx = ax.twiny()  # top ax

        ax.set_yticks(np.arange(0, len(heat_values.index), 1))
        ax.set_yticklabels(
            group_values.iloc[0 : (group_values.shape[0] - 1), 0]
        )  # first column
        ax.set_yticks(np.arange(-0.5, len(heat_values.index), 1), minor=True)

        ax.set_xticks(np.arange(0, len(heat_values.columns), 1))
        ax.set_xticklabels(
            group_values.iloc[
                group_values.shape[0] - 1, 1 : (group_values.shape[1] - 1)
            ]
        )  # last row
        ax.set_xticks(np.arange(-0.5, len(heat_values.columns), 1), minor=True)

        by.set_yticks(np.arange(0, (len(heat_values.index) * 2) + 1, 1))
        right_ticklabels = list(
            group_values.iloc[
                0 : (group_values.shape[0] - 1), group_values.shape[1] - 1
            ]
        )  # last column
        by_ticklabels = [""] * (len(right_ticklabels) * 2 + 1)
        by_ticklabels[1::2] = right_ticklabels
        by.set_yticklabels(by_ticklabels, fontsize=10)

        bx.set_xticks(np.arange(0, (len(heat_values.columns) * 2) + 1, 1))
        top_ticklabels = list(heat_values.columns)
        bx_ticklabels = [""] * (len(top_ticklabels) * 2 + 1)
        bx_ticklabels[1::2] = top_ticklabels
        bx.set_xticklabels(bx_ticklabels, fontsize=10)

        by.invert_yaxis()
        by.autoscale(False)

        ax.grid(which="minor", color="w", linewidth=1)

        plt.show()
