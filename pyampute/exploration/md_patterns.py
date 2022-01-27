"""Some title"""
# Author: Rianne Schouten <r.m.schouten@tue.nl>
# Co-Author: Srinidhi Ilango <s.srinidhi.ilango@student.tue.nl>

from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

Matrix = Union[pd.DataFrame, np.ndarray]


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
        self, X: Matrix, show_plot: bool = True, show_patterns: bool = True
    ) -> Matrix:
        """Some comments

        Parameters
        ----------
        X : Matrix
            Matrix of shape `(n_samples, m_features)`
            Incomplete input data, where "n_samples" is the number of samples and
            "m_features" is the number of features.

        Returns
        -------
        md_patterns: 
        plot_md_patterns: when show_plot is True
        """

        # make sure Y is a pd.DataFrame
        Xdf = pd.DataFrame(X)

        # calculate patterns
        self._calculate_patterns(Xdf)
        if show_patterns:
            print(self.md_patterns)

        # make plot
        if show_plot:
            self._make_plot()

        return self.md_patterns

    def _calculate_patterns(self, X: pd.DataFrame) -> Matrix:
        """
        this function calculates the md patterns
        """

        # mask
        mask = X.isnull()

        # count number of missing values per column
        colsums = mask.sum()
        sorted_col = colsums.sort_values().index.tolist()
        colsums["zero_count"] = colsums.sum()
        colsums["row_count"] = ""

        # finding missing values per group and other required values
        group_values = (~mask).groupby(sorted_col).size().reset_index(name="row_count")
        group_values["zero_count"] = group_values.isin([0]).sum(axis=1)
        group_values.sort_values(
            by=["zero_count", "row_count"], ascending=[True, False], inplace=True
        )
        group_values = group_values.append(colsums, ignore_index=True)

        # put row_count in the begining
        cols = list(group_values)
        cols.insert(0, cols.pop(cols.index("row_count")))
        group_values = group_values.loc[:, cols]

        self.md_patterns = group_values
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
