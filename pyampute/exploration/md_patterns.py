"""Some title"""
# Author: Rianne Schouten <r.m.schouten@tue.nl>
# Co-Author: Srinidhi Ilango <s.srinidhi.ilango@student.tue.nl>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


from pyampute.utils import Matrix


class mdPatterns:
    """
    some comments
    Initialize with nothing, then run get_patterns

    Parameters
    ----------
    incomplete_data : Matrix with shape `(n, m)`
        Dataset with missing values
    missing_values
        The placeholder for the missing values as needed by MissingIndicator

    Methods
    -------
    get_patterns

    Examples
    --------
    X = np.random.randn(100, 3)
    mask1 = np.random.binomial(n=1, size=X.shape[0], p=0.5)    
    mask2 = np.random.binomial(n=1, size=X.shape[0], p=0.5)
    X[mask1==1, 0] = np.nan
    X[mask2==1, 1] = np.nan
    mypat = mp.mdPatterns()
    mdpatterns = mypat.get_patterns(X)  
    """

    def __init__(self):

        # initialize empty attributes
        self.md_patterns = None

    def get_patterns(
        self, X: Matrix, count_or_proportion: str = "count", show_plot: bool = True
    ) -> pd.DataFrame:
        """Some comments

        Parameters
        ----------
        X : Matrix
            Matrix of shape `(n_samples, m_features)`
            Incomplete input data, where "n_samples" is the number of samples and "m_features" is the number of features.
        
        count_or_proportion : str, {"count", "proportion"}
            Whether the patterns should be given in counts or proportions.

        show_plot : bool, default : True
            Whether the patterns should be shown in a plot.

        Returns
        -------
        md_patterns: pd.DataFrame
            A pandas dataframe of shape `(k+2, m_features+2)`
            Here, "k" is the number of patterns, with one extra for rows that do not have missing values and one extra row with column totals, and "m_features" is the number of features, with one extra column for the row_count or row_percent and one extra column for number of missing values per pattern.
        """

        # make sure Y is a pd.DataFrame
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
        """
        Find all unique missing data patterns and structure it as a pd.DataFrame
        """

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
        if group_values.iloc[0,0:-2].values.tolist() != list(np.ones(len(sorted_col))):
            group_values.loc[-1] = np.concatenate((np.ones(len(sorted_col)), np.zeros(2))).astype(int)
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

        group_values = self.md_patterns

        heat_values = group_values.iloc[
            0 : (group_values.shape[0] - 1), 1 : group_values.shape[1] - 1
        ]

        myred = "#B61A51B3"
        myblue = "#006CC2B3"
        cmap = colors.ListedColormap([myred, myblue])

        fig, ax = plt.subplots(1)
        ax.imshow(heat_values, aspect="auto", cmap=cmap)

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
