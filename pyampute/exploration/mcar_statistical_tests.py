"""Statistical hypothesis test for Missing Completely At Random (MCAR)"""

# Author: Rianne Schouten <https://rianneschouten.github.io/>
# Co-Author: Davina Zamanzadeh <https://davinaz.me/>
# Modified to report effect sizes alongside p-values.

from logging import error
import numpy as np
import pandas as pd
from math import pow
from scipy.stats import chi2, ttest_ind

# Local
from pyampute.utils import Matrix


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    """
    Pooled-variance Cohen's d for two samples.

    Returns np.nan when either group has fewer than 2 usable values or the
    pooled standard deviation is 0.
    """
    a = a.dropna()
    b = b.dropna()
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled


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
    def little_mcar_test(X: Matrix, return_stats: bool = False):
        """
        Implementation of Little's MCAR test

        Parameters
        ----------
        X : Matrix of shape `(n, m)`
            Dataset with missing values. `n` rows (samples) and `m` columns (features).

        return_stats : bool, default : False
            If False, returns only the p-value (original behavior). If True,
            returns a dict with the p-value, the chi-square statistic, the
            degrees of freedom, and an effect size.

        Returns
        -------
        pvalue : float
            The p-value of a chi-square hypothesis test when `return_stats` is
            False. Null hypothesis: data is Missing Completely At Random (MCAR).
            Alternative hypothesis: data is not MCAR.

        stats : dict
            Returned when `return_stats` is True, with keys:

            - ``"pvalue"`` : chi-square p-value
            - ``"statistic"`` : Little's chi-square statistic ``d2``
            - ``"df"`` : degrees of freedom
            - ``"effect_size"`` : ``sqrt(d2 / n)``, a Cohen's w style measure.
              Because ``d2`` is a sample-size-weighted sum of Mahalanobis
              distances of the pattern means from the grand mean, this equals
              the root of the average squared Mahalanobis distance and is
              interpretable as a standardized effect magnitude.
        """

        dataset = X.copy()
        vars = dataset.dtypes.index.values
        n_var = dataset.shape[1]
        n_obs = dataset.shape[0]

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

        if return_stats:
            effect_size = np.sqrt(d2 / n_obs) if n_obs > 0 else np.nan
            return {
                "pvalue": pvalue,
                "statistic": float(d2),
                "df": int(df),
                "effect_size": float(effect_size),
            }

        return pvalue

    @staticmethod
    def mcar_t_tests(X: Matrix, effect_size: bool = False):
        """
        Performs t-tests for MCAR for each pair of features.

        Parameters
        ----------
        X : Matrix of shape `(n, m)`
            Dataset with missing values. `n` rows (samples) and `m` columns (features).

        effect_size : bool, default : False
            If False, returns only the p-value matrix (original behavior). If
            True, returns a tuple ``(pvalues, cohens_d)`` where the second
            matrix holds the pooled-variance Cohen's d for each pair.

        Returns
        -------
        pvalues : pandas DataFrame of shape `(m, m)`
            The p-values of t-tests for each pair of features. Null hypothesis for cell :math:`pvalues[h,j]`: data in feature :math:`h` is Missing Completely At Random (MCAR) with respect to feature :math:`j` for all :math:`h,j` in :math:`{1,2,...m}`. Diagonal values do not exist.

        cohens_d : pandas DataFrame of shape `(m, m)`
            Returned when `effect_size` is True. Cell :math:`[h, j]` is Cohen's d
            comparing feature :math:`j` between the rows where feature :math:`h`
            is missing versus present. Positive values mean feature :math:`j` is
            larger on average when :math:`h` is missing.
        """
        dataset = X.copy()
        vars = dataset.dtypes.index.values
        mcar_matrix = pd.DataFrame(
            data=np.zeros(shape=(dataset.shape[1], dataset.shape[1])),
            columns=vars,
            index=vars,
        )
        effect_matrix = pd.DataFrame(
            data=np.full(shape=(dataset.shape[1], dataset.shape[1]), fill_value=np.nan),
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
                effect_matrix.loc[var, tvar] = _cohens_d(part_one, part_two)

        pvalues = mcar_matrix[mcar_matrix.notnull()]

        if effect_size:
            return pvalues, effect_matrix
        return pvalues
