"""Statistical hypothesis test for Missing Completely At Random (MCAR)"""
# Author: Rianne Schouten <https://rianneschouten.github.io/>
# Co-Author: Davina Zamanzadeh <https://davinaz.me/>

from logging import error
import numpy as np
import pandas as pd
from math import pow
from scipy.stats import chi2, ttest_ind

# Local
from pyampute.utils import Matrix


class MCARTest:
    """
    Statistical hypothesis test for Missing Completely At Random (MCAR)

    Performs Little's MCAR test (see `Little, R.J.A. (1988)`_). Null hypothesis: data is Missing Completely At Random (MCAR). Alternative hypothesis: data is not MCAR.

    .. _`Little, R.J.A. (1988)`: https://www.tandfonline.com/doi/abs/10.1080/01621459.1988.10478722

    Parameters
    ----------
    method : str, {"little", "ttest"}, default : "little"
        Whether to perform a chi-square test on the entire dataset ("little") or separate t-tests for every combination of variables ("ttest"). 

    See also
    --------
    :class:`~pyampute.exploration.md_patterns.mdPatterns` : Displays missing data patterns in incomplete datasets
    
    :class:`~pyampute.ampute.MultivariateAmputation` : Transformer for generating multivariate missingness in complete datasets

    Notes
    -----
    We advise to use Little's MCAR test carefully. Rejecting the null hypothesis may not always mean that data is not MCAR, nor is accepting the null hypothesis a guarantee that data is MCAR. See `Schouten et al. (2021)`_ for a thorough discussion of missingness mechanisms. 

    .. _`Schouten et al. (2021)`: https://journals.sagepub.com/doi/full/10.1177/0049124118799376

    Examples
    --------
    >>> import pandas as pd
    >>> from pyampute.exploration.mcar_statistical_tests import MCARTest
    >>> data_mcar = pd.read_table("data/missingdata_mcar.csv")
    >>> mt = MCARTest(method="little")
    >>> print(mt.little_mcar_test(data_mcar))
    0.17365464213775494    
    """

    def __init__(self, method: str = "little"):
        self.method = method

    def __call__(self, data: Matrix) -> float:
        if self.method == "little":
            return self.little_mcar_test(data)
        elif self.method == "ttest":
            return self.mcar_t_tests(data)
        else:
            error(
                f"Chose {self.method} as test method, which is not supported. Please choose from [little, ttest]."
            )

    @staticmethod
    def little_mcar_test(X: Matrix) -> float:
        """
        Implementation of Little's MCAR test
        
        Parameters
        ----------
        X : Matrix of shape `(n, m)`
            Dataset with missing values. `n` rows (samples) and `m` columns (features).

        Returns
        -------
        pvalue : float
            The p-value of a chi-square hypothesis test. Null hypothesis: data is Missing Completely At Random (MCAR). Alternative hypothesis: data is not MCAR.
        """

        dataset = X.copy()
        vars = dataset.dtypes.index.values
        n_var = dataset.shape[1]

        # mean and covariance estimates
        # ideally, this is done with a maximum likelihood estimator
        gmean = dataset.mean()
        gcov = dataset.cov()

        # set up missing data patterns
        r = 1 * dataset.isnull()
        mdp = np.dot(r, list(map(lambda x: pow(2, x), range(n_var))))
        sorted_mdp = sorted(np.unique(mdp))
        n_pat = len(sorted_mdp)
        correct_mdp = list(map(lambda x: sorted_mdp.index(x), mdp))
        dataset["mdp"] = pd.Series(correct_mdp, index=dataset.index)

        # calculate statistic and df
        pj = 0
        d2 = 0
        for i in range(n_pat):
            dataset_temp = dataset.loc[dataset["mdp"] == i, vars]
            select_vars = ~dataset_temp.isnull().any()
            pj += np.sum(select_vars)
            select_vars = vars[select_vars]
            means = dataset_temp[select_vars].mean() - gmean[select_vars]
            select_cov = gcov.loc[select_vars, select_vars]
            mj = len(dataset_temp)
            parta = np.dot(
                means.T, np.linalg.solve(select_cov, np.identity(select_cov.shape[1]))
            )
            d2 += mj * (np.dot(parta, means))

        df = pj - n_var

        # perform test and save output
        pvalue = 1 - chi2.cdf(d2, df)

        return pvalue

    @staticmethod
    def mcar_t_tests(X: Matrix) -> pd.DataFrame:
        """
        Performs t-tests for MCAR for each pair of features.

        Parameters
        ----------
        X : Matrix of shape `(n, m)`
            Dataset with missing values. `n` rows (samples) and `m` columns (features).

        Returns
        -------
        pvalues : pandas DataFrame of shape `(m, m)`
            The p-values of t-tests for each pair of features. Null hypothesis for cell :math:`pvalues[h,j]`: data in feature :math:`h` is Missing Completely At Random (MCAR) with respect to feature :math:`j` for all :math:`h,j` in :math:`{1,2,...m}`. Diagonal values do not exist. 
        """
        dataset = X.copy()
        vars = dataset.dtypes.index.values
        mcar_matrix = pd.DataFrame(
            data=np.zeros(shape=(dataset.shape[1], dataset.shape[1])),
            columns=vars,
            index=vars,
        )

        for var in vars:
            for tvar in vars:
                part_one = dataset.loc[dataset[var].isnull(), tvar].dropna()
                part_two = dataset.loc[~dataset[var].isnull(), tvar].dropna()
                mcar_matrix.loc[var, tvar] = ttest_ind(
                    part_one, part_two, equal_var=False
                ).pvalue

        return mcar_matrix[mcar_matrix.notnull()]
