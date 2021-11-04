"""Transformer for generating multivariate missingness in complete datasets"""
# Author: Rianne Schouten <riannemargarethaschouten@gmail.com>
# Co-Author: Davina Zamanzadeh <davzaman@gmail.com>

from typing import Any, Dict, Tuple, Type, Union
import logging
import numpy as np
from pandas import DataFrame
from sklearn.base import TransformerMixin
from scipy import stats
from math import isclose

# Local
from utils import (
    ArrayLike,
    Matrix,
    isin,
    isnan,
    is_numeric,
    enforce_numeric,
    setup_logging,
    missingness_profile,
    shifted_probability_func,
    standardize_uppercase,
)

# TODO: Odds

# TODO: Add fit and transform separately


class MultivariateAmputation(TransformerMixin):
    """Generating multivariate missingness patterns in complete datasets

    n = number of samples.
    m = number of features/vars.
    k = number of patterns.

    Parameters: <param name> : <type/shape> : <default value>
    ----------
    complete_data : matrix with shape (n, m)
        Dataset with no missing values for vars involved in amputation.
        n rows (samples) and m columns (features).
        Values involved in amputation should be numeric, or will be forced.
        Categorical variables should have been transformed to dummies.

    prop : float [0,1] : 0.5
        Proportion of missingness as a decimal or percent.

    patterns : list of k dictionaries : 1 pattern: {
        "incomplete_vars": random 50% of vars,
        "mechanism": MAR,
        "score-to-prob": sigmoid-right
        "freq": 1
    }
        If there are too many patterns:
            subsequent data subset (for pattern) will be empty, no amputation will occur
        Each dictionary has the following key-value pairs (required unless [optional]):
        Note: the required entries are required for ALL patterns
            (the defaults will not extend to missing entries, unless optional).
        - incomplete_vars : list-like of var/col {indices (ints) or names (strs)}
                          : array of 50% randomly chosen indices (range 0:m-1)
            Indicates variables that should be amputed.
            observed_vars = list-like(complement of incomplete_vars).
        - weights [optional]
                : list-like of m floats [0,1] OR
                  dict {index/name: weight} for each var influencing missingness
                : if mechanism == MCAR: all 0's
                                  MAR: observed_vars weight 1
                                  MNAR: incomplete_vars weight 1.
            Specify size of effect of each specified var on missing vars.
                - negative (decrease effect)
                - 0 (no role in missingness) If dict, unspecified vars have weight 0.
                - positive (increase effect).
            Missing score for sample i in pattern k = innerproduct(weights, sample[i]).
        - mechanism : string : MAR
            Choices: [MCAR, MAR, MNAR, MAR+MNAR] case insensitive.
        - freq : float [0,1] : all patterns with equal frequency (1/k)
            Relative occurence of a pattern with respect to other patterns.
            All frequencies across k dicts/patterns must sum to 1.
            For example (k = 3 patterns), freq := [0.4, 0.4, 0.2] =>
                of all samples with missing values,
                40% should have pattern 1, 40% pattern 2. and 20% pattern 3.
        - score_to_probability_func
            : str or fn[list-like floats [-inf, inf] -> list-like floats [0, 1]]
            : sigmoid-right
            Converts standardized weighted scores for each sample
                (in a data subset corresponding to pattern k)
                to probability of missingness.
            Choices for string: sigmoid-[RIGHT, LEFT, MID, TAIL], case insensitive.
                Applies sigmoid function with a logit cutoff per pattern.
                Dictates a [high, low, average, extreme] score
                    (respectively) has a high probability of amputation.
            Fn will be shifted to ensure correct joint missingness probabilities.

    std : boolean : True
        Whether or not to standardize data before computing scores.
        Don't standardize if passing both train and test (prevent leaking).

    lower_range : float : -3
        Lower limit in range to search for b, the horizontal shift
        of the inputs to the sigmoid function in order to assign
        a probability for a value to be missing.

    upper_range : float : 3
        Upper limit in range to search for b, the horizontal shift
        of the inputs to the sigmoid function in order to assign
        a probability for a value to be missing.

    max_dif_with_target : float : 0.001
        The allowable error between the desired percent missing data (prop)
        and and calculated joint missing probability after assigning a
        probability for values to be missing.

    max_iter : integer : 100
        Max number of iterations for binary search when searching for b,
        the horizontal shift of the inputs (weighted sum scores) to the
        sigmoid function.


    Attributes
    ----------
    incomplete_data :  matrix with shape (n, m)
        Dataset with missing values.

    Notes
    -----
    Something on difference ampute in R and Python
    #TODO: any more detailed explanations

    References
    ----------
    .. [1] Rianne Margaretha Schouten, Peter Lugtig & Gerko Vink (2018).
    Generating missing values for simulation purposes:
        A multivariate amputation procedure.
    Journal of Statistical Computation and Simulation, DOI:
        10.1080/00949655.2018.1491577
    """

    def __init__(
        self,
        prop: float = 0.5,
        patterns: Dict[str, Any] = None,
        std: bool = True,
        lower_range: float = -3,
        upper_range: float = 3,
        max_dif_with_target: float = 0.001,
        max_iter: int = 100,
    ):
        self.prop = prop
        self.patterns = patterns
        self.std = std
        self.lower_range = lower_range
        self.upper_range = upper_range
        self.max_dif_with_target = max_dif_with_target
        self.max_iter = max_iter
        # The rest are set by _pattern_dict_to_matrix_form()

        setup_logging()

    def _binary_search(
        self, wss_standardized: ArrayLike, pattern_ind: int
    ) -> Tuple[float, Matrix]:
        """
        Search for the appropriate shift/transformation to the scores before passing
            through the self.probability_function to result in the desired missingness
            proportion.  e.g. raw wss will mask 17% of samples in pattern k but you want
            40% missing.
        """

        b = 0
        counter = 0
        lower_range = self.lower_range
        upper_range = self.upper_range

        probs_matrix = None

        # start binary search with a maximum amount of tries of max_iter
        while counter < self.max_iter:
            counter += 1

            # in every iteration, the new b is the mid of the lower and upper range
            # the lower and upper range are updated at the end of each iteration
            b = lower_range + (upper_range - lower_range) / 2
            if counter == self.max_iter:
                break

            # calculate the expected missingness proportion
            # depends on the logit cutoff type, the sum scores and b
            probs_matrix = shifted_probability_func(
                wss_standardized, b, self.score_to_probability_func[pattern_ind]
            )
            current_prop = np.mean(probs_matrix)

            # if the expected proportion is close to the target, break
            # the maximum difference can be specified
            # if max_dif_with_target is 0.001, the proportion differs with max 0.1%
            if np.absolute(current_prop - self.prop) < self.max_dif_with_target:
                break

            # if we have not reached the desired proportion
            # we adjust either the upper or lower range
            # this way works for self.score_to_probability_func[i] = 'SIGMOID-RIGHT'
            # TODO: discuss about this working for custom and other ways
            # need to check for the other types
            # in the next iteration, a new b is then calculated and used
            if (current_prop - self.prop) > 0:
                upper_range = b
            else:
                lower_range = b

        return b, probs_matrix

    def _choose_probabilities(self, wss: ArrayLike, pattern_index: int) -> Matrix:
        """
        Assigns missingness probabilities for each sample in the data subset
            corresponding to pattern k (pattern_index) using the standardized wss.
        This is later thresholded to use to decide whether or not to apply pattern k
        to sample i.

        """
        # when wss contains merely zeros, the mechanism is
        # 1. MCAR: each case has an equal probability of becoming missing
        # 2. MAR with binary variables
        # Therefore we just use uniform probability of missing per var using self.freqs
        if np.all(wss == 0):
            probs = np.repeat(self.freqs[pattern_index], len(wss))
        else:  # else we calculate the probabilities based on the wss
            # standardize wss
            wss_standardized = stats.zscore(wss)
            # calculate the size of b for the desired missingness proportion
            b, probs_matrix = self._binary_search(wss_standardized, pattern_index)
            probs = np.squeeze(np.asarray(probs_matrix))

        return probs

    def _calculate_sumscores(self, data_group: Matrix, pattern_ind: int) -> ArrayLike:
        """
        Creates a vector of weighted sum score for each sample in the data subset
        corresponding to pattern k by computing the inner product of
            self.weights and the raw values of the samples in that subset.

        This is later converted to a probability to be thresholded on to decide
            whether or not to apply pattern k to sample i in the data subset.
        """

        # transform only vars involved in amputation to numeric to compute weights
        # does not transform the original datset
        logging.info(
            "Enforcing data to be numeric since calculation of weights"
            " requires numeric data."
        )
        data_group = enforce_numeric(data_group)
        # standardize data or not
        if self.std:
            data_group = stats.zscore(data_group)

        # calculate sum scores
        # in case of MCAR, weights[i, ] contains merely zeros and wss are merely zeros
        # in case of MAR, MNAR, the mechanisms is determined by the weights
        wss = np.dot(data_group, self.weights[pattern_ind, :].T)

        return wss

    def _get_default_pattern(self, m_features: int) -> Dict[str, Any]:
        return {
            # Random half of vars (random 50% of indices)
            "incomplete_vars": np.random.choice(
                np.arange(m_features), int(m_features / 2), replace=False
            ),
            "mechanism": "MAR",
            "freq": 1,
            "score_to_probability_func": "sigmoid-RIGHT",
        }

    def _pattern_dict_to_matrix_form(self):
        """
        Converts the list of dictionaries into corresponding matrices and arrays.
        Each dict entry that's list-like will transform into a matrix (k, m)
            e.g., weight array for pattern i will define the row i in the weight matrix
        Each dict entry that's a single value will transform into an array of length m
            e.g., freq for pattern i will define ith entry in freqs array.
        """
        k_by_m = (self.num_patterns, self.num_features)

        #### Init ####
        # indicator matrix (k, m) {0, 1}, (previously called patterns)
        self.observed_var_indicator = np.empty(shape=k_by_m, dtype=bool)
        # weight for scores matrix (k, m) [-inf, inf]
        self.weights = np.empty(shape=k_by_m, dtype=float)
        # array of mechanisms per pattern (len k)
        self.mechanisms = np.empty(shape=self.num_patterns, dtype=str)
        # array of frequencies per pattern (len k)
        self.freqs = np.empty(shape=self.num_patterns, dtype=float)
        # list of functions or strings per pattern (len k)
        self.score_to_probability_func = []

        def populate_pattern_array(
            indices_or_names: ArrayLike,
            fill_value: Union[float, ArrayLike],
            dtype: Type,
        ) -> ArrayLike:
            """
            Fills an array of length m (for each feature) with fill_value
                wherever indicated by indices_or_names.
            Column names will be mapped to their corresponding indices.
            """
            # init zeros so unmentioned vars have no effect
            matrix_row_entry = np.zeros(shape=self.num_features, dtype=dtype)
            # if names, convert to indices
            if isinstance(indices_or_names, str):
                indices = [self.colname_to_idx[coln] for coln in indices_or_names]
            # force to np array to act as indexer
            indices = np.array(indices_or_names)
            # if int will fill same int for all indices
            # else len(indices) == len(fill_value)
            matrix_row_entry[indices] = fill_value
            return matrix_row_entry

        #### Build from Dicts ####
        for i, pattern in enumerate(self.patterns):
            # one-hot the at corresponding indices
            amputed_var_indicator = populate_pattern_array(
                pattern["incomplete_vars"], fill_value=1, dtype=bool
            )
            # flip indicator
            self.observed_var_indicator[i] = 1 - amputed_var_indicator

            # TODO: add checks that the dict values make sense?
            if "weights" in pattern:
                if isinstance(pattern["weights"], Dict):
                    # basically unzip dictionary into 2 lists of equal length
                    indices_or_names, weights_per_var = zip(*pattern["weights"].items())
                    self.weights[i] = populate_pattern_array(
                        indices_or_names, fill_value=weights_per_var, dtype=float
                    )
                else:  # array of weights directly given
                    self.weights[i] = pattern["weights"]
            else:  # weights missing, fill with nan
                self.weights[i] = np.full(shape=self.num_features, fill_value=np.nan)

            self.mechanisms[i] = pattern["mechanism"]
            self.freqs[i] = pattern["freq"]
            self.score_to_probability_func.append(pattern["score_to_probability_func"])

    def _set_defaults(self):
        """
        Set defaults for args, assuming patterns has been initialized.
        Most of the defaults rely on info from patterns.
        Will adjust vars:
            change % to decimal, repeat for all patterns,
            standardize strings to uppercase force lists to np arrays, etc.)
        """
        # check for prop that makes sense, since we validate after setting defaults
        if self.prop > 1 and self.prop <= 100:
            logging.info(
                "Detected proportion of missingness to be percentage,"
                " converting to decimal."
            )
            self.prop /= 100

        # RELIES ON: patterns
        # force numpy
        self.freqs = np.array(self.freqs)
        # TODO: recalculate frequencies to sum to 1?

        # RELIES ON: patterns
        # just standardize to upper case
        self.mechanisms = np.array(list(map(standardize_uppercase, self.mechanisms)))

        # RELIES ON: patterns
        # just standardize to upper case if string
        self.score_to_probability_func = np.array(
            [
                standardize_uppercase(func) if isinstance(func, str) else func
                for func in self.score_to_probability_func
            ]
        )

        # RELIES ON: patterns, mechanisms
        # if only some weights are indicated:
        #   fill the rest (all nan rows) with default according to mechanism.
        if np.isnan(self.weights).any(axis=None):
            logging.info(
                "No weights passed for some patterns, filling them in per pattern."
                " MCAR: weights are all 0s."
                " MAR: all observed vars have weight 1."
                " MNAR: all missing vars have weight 1."
            )
            patterns_with_missing_weights = np.isnan(self.weights).all(axis=1)

            self.weights[
                patterns_with_missing_weights & self.mechanisms == "MCAR"
            ] = np.zeros(shape=self.num_features)

            missing_mar_mask = patterns_with_missing_weights & self.mechanisms == "MAR"
            self.weights[missing_mar_mask] = self.observed_var_indicator[
                missing_mar_mask
            ]

            missing_mnar_mask = (
                patterns_with_missing_weights & self.mechanisms == "MNAR"
            )
            # note that non-observed is given a value 0 in indicator matrix
            self.weights[missing_mnar_mask] = (
                1 - self.observed_var_indicator[missing_mnar_mask]
            )

    def _validate_args(self):
        """
        Validates remainined constructor args after having set defaults.
        Only makes assertions, assuming everything is initialized.
        """
        ##################################
        #     OBSERVED VAR INDICATOR     #
        ##################################
        # axis=None reduces all axes for both pandas and numpy
        assert isin(self.observed_var_indicator, [0, 1]).all(
            axis=None
        ), "Indicator matrix can only contain 0's and 1's."
        assert not (
            (self.observed_var_indicator == 1).all(axis=None)
        ), "Cannot indicate no features to be amputed, will result in no amputation."
        if isin(self.mechanisms, "MAR").any(axis=0):
            assert not (self.observed_var_indicator[self.mechanisms == "MAR"] == 0).all(
                axis=None
            ), "Cannot ampute all features under MAR, since all vars will be missing."

        ##################
        #      PROP      #
        ##################
        assert self.prop >= 0 and self.prop <= 100, (
            "Proportion of missingness should be a value between 0 and 1"
            " (for a proportion) or between 1 and 100 (for a percentage)"
        )

        ###################
        #   FREQUENCIES   #
        ###################
        assert len(self.freqs) == self.num_patterns, (
            "There should be a frequency of missingness for every pattern,"
            f" but there are only {len(self.freqs)} frequencies specified,"
            f" and {self.num_patterns} patterns specified from `patterns`."
        )
        assert (self.freqs >= 0).all() and (
            self.freqs <= 1
        ).all(), "Frequencies must be between 0 and 1 inclusive."
        # there's imprecision in float, so it might be 0.9999999
        assert isclose(sum(self.freqs), 1), "Frequencies should sum to 1."

        ##################
        #   MECHANISMS   #
        ##################
        assert (
            len(self.mechanisms) == self.num_patterns
        ), "Must specify a mechanism per pattern, but they do not match."
        mechanism_options = ["MCAR", "MAR", "MNAR", "MAR+MNAR"]
        assert isin(
            self.mechanisms, mechanism_options
        ).all(), f"Mechanisms specified must be one of {mechanism_options}."

        #################
        #    WEIGHTS    #
        #################
        assert (
            self.weights.shape == self.observed_var_indicator.shape
        ), "Weights passed must match dimensions of patterns passed."
        assert (self.weights[self.mechanisms == "MCAR"] == 0).all(
            axis=None
        ), "Patterns with MCAR should have weights of all 0's."
        # TODO: add checks if weights make sense with coresponding mechanisms
        # TODO: cannot pass MAR+MNAR with no weights

        #####################################
        #     SCORE TO PROBABILITY FUNC     #
        #####################################
        assert (
            len(self.score_to_probability_func) == self.num_patterns
        ), "Types, mechs, and freqs must all be the same dimension (# patterns)."
        func_str_options = (
            ["SIGMOID-RIGHT", "SIGMOID-LEFT", "SIGMOID-MID", "SIGMOID-TAIL"],
        )
        # check only the str entries in the funcs list
        assert isin(
            [fn for fn in self.score_to_probability_func if isinstance(fn, str)],
            func_str_options,
        ).all(), f"String funcs can only be one of {func_str_options}"

    def _validate_input(self, X: Matrix) -> Matrix:
        """
        Validates input data with given arguments to amputer.
        Will modify the dataset to comply if possible, while giving warnings.
        """
        # This must come first so we can check patterns
        assert X is not None, "No dataset passed, cannot be None."
        assert len(X.shape) == 2, "Dataset must be 2 dimensional."
        num_features = X.shape[1]

        ##################
        #    PATTERNS    #
        ##################
        if self.patterns is None:
            logging.info("No patterns passed, setting default pattern.")
            self.patterns = self._get_default_pattern(num_features)
        assert len(self.patterns) > 0, "Must contain at least one pattern."
        # check each dict has the required entries (via superset check)
        required_keys = {
            "incomplete_vars",
            "mechanism",
            "freq",
            "score_to_probability_func",
        }
        assert {keys for keys, _ in self.patterns.items()}.issuperset(required_keys), (
            "Patterns is malformed. "
            f"Each dict in the list must contain at least these keys: {required_keys}. "
            "The 'weights' entry is optional."
        )

        # bookkeeping vars for readability
        self.num_patterns = len(self.patterns)
        self.num_features = num_features
        self.colname_to_idx = {colname: idx for idx, colname in enumerate(X.columns)}

        # converts patterms to matrix form for easy interal processing
        self._pattern_dict_to_matrix_form()
        # defaults for the rest of the args (depends on patterns being initialized)
        self._set_defaults()
        self._validate_args()

        # vars involved in amputation have scores computed and need to be
        #   complete and numeric
        # A var (column) is involved if for any pattern (row) it has a weight.
        vars_involved_in_ampute = (self.weights != 0).any(axis=0)

        ##################
        #      DATA      #
        ##################
        assert X.shape[1] > 1, "Dataset passed must contain at least two columns."
        # enforce numpy just for checking
        X_check = X.values if isinstance(X, DataFrame) else X
        assert not isnan(
            X_check[:, vars_involved_in_ampute]
        ).any(), "Features involved in amputation must be complete, but contains NaNs."
        if not is_numeric(X_check[:, vars_involved_in_ampute]):
            logging.warn(
                "Features involved in amputation found to be non-numeric."
                " They will be forced to numeric upon calculating sum scores."
            )

        return X

    def fit_transform(self, X: Matrix) -> Matrix:
        """Fits amputer on complete data X and returns the incomplete data X

        Parameters
        ----------
        X : matrix of shape (n_samples, m_features)
            Complete input data, where "n_samples" is the number of samples and
            "m_features" is the number of features.

        Returns
        -------
        X_incomplete : matrix of shape (n_samples, m_features)
        """

        # sets defaults, adjusts vars, and runs checks
        X = self._validate_input(X)

        # split complete_data in groups
        # the number of groups is defined by the number of patterns
        num_samples = X.shape[0]
        X_incomplete = np.zeros(X.shape)
        X_indices = np.arange(num_samples)
        assigned_group_number = np.random.choice(
            a=self.num_patterns, size=num_samples, p=self.freqs
        )

        # start a loop over each pattern
        for pattern_ind in range(self.num_patterns):
            # assign cases to the group
            group_indices = X_indices[assigned_group_number == pattern_ind]
            pattern = np.squeeze(
                np.asarray(self.observed_var_indicator[pattern_ind, :])
            )
            data_group = X[group_indices]
            # calculate weighted sum scores for each sample in the group
            wss = self._calculate_sumscores(data_group, pattern_ind)
            # define candidate probabilities in group
            probs = self._choose_probabilities(wss, pattern_ind)
            # apply probabilities and choose cases
            chosen_candidates = np.random.binomial(
                n=1, size=data_group.shape[0], p=probs
            )
            # apply missing data pattern
            chosen_indices = group_indices[chosen_candidates == 1]
            X_incomplete[chosen_indices, pattern == 0] = np.nan

        missingness_profile(X_incomplete)
        return X_incomplete
