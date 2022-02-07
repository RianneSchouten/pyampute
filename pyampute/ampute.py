"""Transformer for generating multivariate missingness in complete datasets"""
# Author: Rianne Schouten <riannemargarethaschouten@gmail.com>
# Co-Author: Davina Zamanzadeh <davzaman@gmail.com>

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import logging
import numpy as np
from pandas import DataFrame, read_csv, isnull
from sklearn.base import TransformerMixin
from scipy import stats
from math import isclose

# Local
from pyampute.utils import (
    LOOKUP_TABLE_PATH,
    ArrayLike,
    Matrix,
    isin,
    is_numeric,
    enforce_numeric,
    setup_logging,
    standardize_uppercase,
    sigmoid,
)

THRESHOLD_MIN_NUM_CANDIDATES = 10
THRESHOLD_MIN_NUM_UNIQUE_WSS = 5


class MultivariateAmputation(TransformerMixin):
    """Generating multivariate missingness patterns in complete datasets

    - `n` = number of samples.
    - `m` = number of features/vars.
    - `k` = number of patterns.

    Read more about this and this here. And put reference somewhere.

    Parameters
    ----------
    complete_data : Matrix with shape `(n, m)`
        Dataset with no missing values for vars involved in amputation.
        `n` rows (samples) and `m` columns (features).
        Values involved in amputation should be numeric, or will be forced, and any columns that aren't fully numeric will be dropped.

    prop : float, default : 0.5
        Proportion of missingness as a decimal or percent.

    patterns : List[Dict], default: ``DEFAULT_PATTERN``
        List of `k` dictionaries.
        If there are too many patterns, the subsequent data subset (for pattern) will be empty, and *no amputation will occur*.
        Each dictionary has the following key-value pairs (required unless [optional]):

            **incomplete_vars** (Union[ArrayLike[int], ArrayLike[str]]) --
                Indicates variables that should be amputed.
                List of int for indices of variables, list of str for column names of variables.
                ``observed_vars`` is the complement of ``incomplete_vars``.

            **weights** (Union[ArrayLike[float], Dict[int, float], Dict[str, float]], default: all 0s (MCAR) or `observed_vars` weight 1 (MAR) or `incomplete_vars` weight 1 (MNAR)) --
                Specifies the size of effect of each specified var on missing vars.
                If using an array, you must specify all *m* weights.
                If using a dictionary, the keys are either indices of vars or column names; unspecified vars will be assumed to have a weight of 0.
                Negative values have a decrease effect, 0s indicate no role in missingness, unspecified vars have weight 0), and positive values have an increase effect.
                The weighted score for sample `i` in pattern `k` is the inner product of the `weights` and `sample[i]`.
                **Note:** weights are required to be defined if the corresponding mechanism is MAR+MNAR.

            **mechanism** (`str, {MAR, MCAR, MNAR, MAR+MNAR} case insensitive`) --
                MNAR+MAR is only possible by passing a custom weight array.

            **freq** (*float [0,1], default: all patterns with equal frequency (1/k)*) --
                Relative occurence of a pattern with respect to other patterns.
                All frequencies across `k` dicts/patterns must sum to 1.
                Either specify for all patterns, or none for the default.
                For example (`k` = 3 patterns), ``freq := [0.4, 0.4, 0.2]`` means, of all samples with missing values, 40% should have pattern 1, 40% pattern 2. and 20% pattern 3.

            **score_to_probability_func** (`Union[str, Callable[ArrayLike[floats] -> ArrayLike[floats]]], {"sigmoid-right", "sigmoid-left", "sigmoid-mid", "sigmoid-tail", Callable}`) --
                Converts standardized weighted scores for each sample (in a data subset corresponding to pattern k) to probability of missingness.
                Choosing one of the sigmoid options (case insensitive) applies sigmoid function with a logit cutoff per pattern.
                The simgoid function will dictates that a [high, low, average, extreme] score (respectively) has a high probability of amputation.
                The sigmoid functions will be shifted to ensure correct joint missingness probabilities.
                Custom functions must accept arrays with values ``(-inf, inf)`` and output values ``[0,1]``.
                We will *not* shift custom functions, refer to :ref:`sphx_glr_auto_examples_plot_custom_probability_function.py` for more.

    std : boolean, default : True
        Whether or not to standardize data before computing weighted scores.
        Standardization ensures that weights can be applied properly.
        Do not standardize if passing both train and test set (prevent leaking).

    lower_range : float, default : -3
        Lower limit in range when searching for horizontal shift of `score_to_probability_func`.

    upper_range : float, default : 3
        Upper limit in range when searching for horizontal shift of `score_to_probability_func`.

    max_dif_with_target : float, default : 0.001
        The allowable error between the desired percent missing data (prop)
        and calculated joint missingness probability after assigning a
        probability for cases to be missing.

    max_iter : int, default : 100
        Max number of iterations for binary search when searching for horizontal shift of `score_to_probability_func`.

    seed: int, optional
        If you want reproducible results during amputation set an integer seed.
        If you don't set it, a random number will be produced every time.

    Attributes
    ----------
    DEFAULT_PATTERN: Dict[str, Any]
        If patterns is not passed, the default is the following:
        .. code-block::
            :caption: Default pattern.

            {
                "incomplete_vars": random 50% of vars,
                "mechanism": "MAR",
                "score-to-prob": "sigmoid-right"
                "freq": 1
            }
   
    DEFAULTS :  Dict[str, str]
        Default values used, especially if values are not passed for arguments in certain patterns (not to be confused with patterns not being specified at all).

    Methods
    -------
    fit_transform(X)

    Examples
    --------
    TODO
    """

    DEFAULTS = {
        "score_to_probability_func": "SIGMOID-RIGHT",
        "mechanism": "MAR",
        "lower_range": -3,
        "upper_range": 3,
        "max_iter": 100,
        "max_diff_with_target": 0.001,
    }

    def __init__(
        self,
        prop: float = 0.5,
        patterns: Dict[str, Any] = None,
        std: bool = True,
        lower_range: float = -3,
        upper_range: float = 3,
        max_diff_with_target: float = 0.001,
        max_iter: int = 100,
        seed: Optional[int] = None,
    ):
        self.prop = prop
        self.patterns = patterns
        self.std = std
        self.lower_range = lower_range
        self.upper_range = upper_range
        self.max_diff_with_target = max_diff_with_target
        self.max_iter = max_iter
        self.seed = seed

        # The rest are set by _pattern_dict_to_matrix_form()
        setup_logging()

    @staticmethod
    def _shifted_probability_func(
        wss_standardized: ArrayLike,
        shift_amount: float,
        probability_func: Union[str, Callable[[ArrayLike], ArrayLike]],
    ) -> ArrayLike:
        """
        Applies shifted custom function or sigmoid (with cutoff) to
            standardized weighted sum scores to convert to probability.
        String: sigmoid-
            Right: Regular sigmoid pushes larger values to have high probability,
            Left: To flip regular sigmoid across y axis, make input negative.
                This pushes smaller values to have high probability.
            Mid: Values in the center of the score distribution have high probability.
            Tail: Larger and smaller values have high probabiliy.    
        """

        if isinstance(probability_func, str):
            cutoff_transformations = {
                "SIGMOID-RIGHT": lambda wss_standardized, b: wss_standardized + b,
                "SIGMOID-LEFT": lambda wss_standardized, b: -wss_standardized + b,
                "SIGMOID-TAIL": lambda wss_standardized, b: (
                    np.absolute(wss_standardized) - 0.75 + b
                ),
                "SIGMOID-MID": lambda wss_standardized, b: (
                    -np.absolute(wss_standardized) + 0.75 + b
                ),
            }

            return sigmoid(
                cutoff_transformations[probability_func](wss_standardized, shift_amount)
            )
        return probability_func(wss_standardized) + shift_amount

    @staticmethod
    def _binary_search(
        wss_standardized: ArrayLike,
        score_to_probability_func: Union[str, Callable[[ArrayLike], ArrayLike]],
        missingness_percent: float,
        lower_range: float,
        upper_range: float,
        max_iter: int,
        max_diff_with_target: float,
    ) -> Tuple[float, ArrayLike]:
        """
        Search for the appropriate shift/transformation to the scores before passing
            through the self.probability_function to result in the desired missingness
            proportion. For instance, raw wss will mask 17% of samples in pattern k but you want
            40% missing.
        """

        b = 0
        counter = 0
        probs_array = None

        # start binary search with maximum number of iterations of max_iter
        while counter < max_iter:
            counter += 1

            # in every iteration, the new b is the mid of the lower and upper range
            # the lower and upper range are updated at the end of each iteration
            b = lower_range + (upper_range - lower_range) / 2
            if counter == max_iter:
                break

            # calculate the expected missingness proportion
            # depends on the logit cutoff type, the sumscores and b
            probs_array = MultivariateAmputation._shifted_probability_func(
                wss_standardized, b, score_to_probability_func
            )
            current_prop = np.mean(probs_array)

            # if the expected proportion is close to the target, break
            # the maximum difference can be specified
            # if max_dif_with_target is 0.001, the proportion differs with max 0.1%
            if np.absolute(current_prop - missingness_percent) < max_diff_with_target:
                break

            # if we have not reached the desired proportion
            # we adjust either the upper or lower range
            # this way works for self.score_to_probability_func[i] = 'SIGMOID-RIGHT'
            # need to check for the other types
            # in the next iteration, a new b is then calculated and used
            if (current_prop - missingness_percent) > 0:
                upper_range = b
            else:
                lower_range = b

        return b, probs_array

    def _calculate_probabilities_from_wss(
        self,
        wss_standardized: ArrayLike,
        score_to_probability_func: Union[str, Callable[[ArrayLike], ArrayLike]],
        missingness_percent: float,
        lower_range: float,
        upper_range: float,
        max_iter: int,
        max_diff_with_target: float,
    ) -> ArrayLike:
        if isinstance(score_to_probability_func, str):
            if self.shift_lookup_table is not None:
                logging.info(
                    "Rounding proportion of missingness to 2 decimal places in order to use lookup table for one of the prespecified score to probability functions."
                )
                prop = np.around(self.prop, 2)

                shift = self.shift_lookup_table.loc[
                    score_to_probability_func, str(prop)
                ]
                return self._shifted_probability_func(
                    wss_standardized, shift, score_to_probability_func
                )
            # If no lookup table, but sigmoid, run binary search
            return self._binary_search(
                wss_standardized,
                score_to_probability_func,
                missingness_percent,
                lower_range,
                upper_range,
                max_iter,
                max_diff_with_target,
            )[1]

        # if not sigmoid, no binary search/shift
        return score_to_probability_func(wss_standardized)

    def _choose_probabilities(self, wss: ArrayLike, pattern_index: int) -> ArrayLike:
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
            probs_array = self._calculate_probabilities_from_wss(
                wss_standardized,
                self.score_to_probability_func[pattern_index],
                self.prop,
                self.lower_range,
                self.upper_range,
                self.max_iter,
                self.max_diff_with_target,
            )
            probs = np.squeeze(np.asarray(probs_array))

        return probs

    def _calculate_sumscores(self, data_group: Matrix, pattern_ind: int) -> ArrayLike:
        """
        Creates a vector of weighted sum scores for each sample in the data subset
        corresponding to pattern k by computing the inner product of
            self.weights and the raw values of the samples in that subset.

        This is later converted to a probability to be thresholded on to decide
            whether or not to apply pattern k to sample i in the data subset.
        """

        if len(data_group) <= THRESHOLD_MIN_NUM_CANDIDATES:
            logging.warn(
                f"Subset for pattern {pattern_ind} is small. "
                "Too many patterns can result in subsets with 0 or few candidates. "
                "Subsets with 0 candidates will be skipped. "
                "Under MCAR, subsets with few candidates will be amputed as normal."
            )

        # transform only vars involved in amputation to numeric to compute weights
        # does not transform the original datset
        data_group = enforce_numeric(data_group, self.vars_involved_in_ampute)
        # standardize data or not
        if self.std:
            data_group = stats.zscore(data_group)

        # calculate sum scores
        # in case of MCAR, weights[i, ] contains merely zeros and wss are merely zeros
        # in case of MAR, MNAR, the mechanisms is determined by the weights
        wss = np.dot(data_group, self.weights[pattern_ind, :].T)

        if len(np.unique(wss)) <= THRESHOLD_MIN_NUM_UNIQUE_WSS:
            logging.warning(
                f"Candidates for pattern {pattern_ind} all have almost the same weighted sum scores. "
                "It is possible this is due to the use of binary variables in amputation. "
                "This creates problems when using the sigmoid function for the score_to_probability_func. "
                "Currently our solution is as follows: if there is just one candidate with a sum score 0, we will ampute it. "
                "If there is one candidate with a nonzero sum score, or multiple candidates with the same score, we evenly apply the same amount of missingness (as if MCAR)."
            )
        return wss

    def _get_default_pattern(self, m_features: int) -> List[Dict[str, Any]]:
        """Default pattern is a single pattern that works for any dataset."""
        # set seed for choice, if None it will be random.
        np.random.seed(self.seed)
        return [
            {
                # Random half of vars (random 50% of indices)
                "incomplete_vars": np.random.choice(
                    np.arange(m_features), int(m_features / 2), replace=False
                ),
                "mechanism": "MAR",
                "freq": 1,
                "score_to_probability_func": "sigmoid-RIGHT",
            }
        ]

    def _validate_indices_or_names(self, indices_or_names: ArrayLike, pattern_idx: int):
        """Validation of dict entries that are lists to be slotted into matrices."""
        assert len(indices_or_names) <= self.num_features, (
            "Cannot list more columns than there exist in the data "
            f"(pattern {pattern_idx})."
        )

        if isinstance(indices_or_names[0], str):
            assert set(self.colname_to_idx.keys()).issuperset(indices_or_names), (
                "One or more column names listed is not a column in the provided data"
                f"(pattern {pattern_idx}"
            )
        else:  # indices, assumes np array
            assert all(indices_or_names >= 0) and all(
                indices_or_names < self.num_features
            ), (
                "One or more indices listed is incorrect (outside range) "
                f"(pattern {pattern_idx}"
            )

    def _populate_pattern_array(
        self,
        indices_or_names: ArrayLike,
        fill_value: Union[float, ArrayLike],
        dtype: Type,
        pattern_idx: int,
    ) -> ArrayLike:
        """
        Fills an array of length m (for each feature) with fill_value
            wherever indicated by indices_or_names.
        Column names will be mapped to their corresponding indices.
        """
        # init zeros so unmentioned vars have no effect
        matrix_row_entry = np.zeros(shape=self.num_features, dtype=dtype)
        # force to np array to act as indexer
        indices_or_names = np.array(indices_or_names)

        # no incomplete vars passed, ignore
        if indices_or_names is None or len(indices_or_names) == 0:
            return matrix_row_entry

        self._validate_indices_or_names(indices_or_names, pattern_idx=pattern_idx)
        # if names, convert to indices
        if isinstance(indices_or_names[0], str):
            indices_or_names = [self.colname_to_idx[coln] for coln in indices_or_names]
        # if int will fill same int for all indices
        # else len(indices) == len(fill_value)
        matrix_row_entry[indices_or_names] = fill_value
        return matrix_row_entry

    def _pattern_dict_to_matrix_form(self):
        """
        Converts the list of dictionaries into corresponding matrices and arrays.
        Each dict entry that's ArrayLike will transform into a matrix (k, m)
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
        # dtype obj instead of str or else only one char will be assigned
        self.mechanisms = np.empty(shape=self.num_patterns, dtype=object)
        # array of frequencies per pattern (len k)
        self.freqs = np.full(
            shape=self.num_patterns, fill_value=1 / self.num_patterns, dtype=float
        )
        # list of functions or strings per pattern (len k)
        self.score_to_probability_func = []

        #### Build from Dicts ####
        for pattern_idx, pattern in enumerate(self.patterns):
            # one-hot the at corresponding indices
            amputed_var_indicator = self._populate_pattern_array(
                pattern["incomplete_vars"],
                fill_value=1,
                dtype=bool,
                pattern_idx=pattern_idx,
            )
            # flip indicator
            self.observed_var_indicator[pattern_idx] = 1 - amputed_var_indicator

            if "weights" in pattern:
                if isinstance(pattern["weights"], Dict):
                    # basically unzip dictionary into 2 lists of equal length
                    indices_or_names, weights_per_var = zip(*pattern["weights"].items())
                    self.weights[pattern_idx] = self._populate_pattern_array(
                        indices_or_names,
                        fill_value=weights_per_var,
                        dtype=float,
                        pattern_idx=pattern_idx,
                    )
                else:  # array of weights directly given
                    self.weights[pattern_idx] = pattern["weights"]
            else:  # weights missing, fill with nan
                self.weights[pattern_idx] = np.full(
                    shape=self.num_features, fill_value=np.nan
                )

            #### Single default values if not specified ####
            self.mechanisms[pattern_idx] = (
                pattern["mechanism"]
                if "mechanism" in pattern
                else self.DEFAULTS["mechanism"]
            )
            self.score_to_probability_func.append(
                pattern["score_to_probability_func"]
                if "score_to_probability_func" in pattern
                else self.DEFAULTS["score_to_probability_func"]
            )

            #### All or None ####
            if "freq" in pattern:
                self.freqs[pattern_idx] = pattern["freq"]

    def _load_shift_lookup_table(self):
        """
        Read the lookup table csv from path.
        Loads the table only once for the shift lookup when computing missing probabilities from scores.
        This is only useful for prespecified functions (e.g., sigmoid-RIGHT)
        """
        if any([isinstance(func, str) for func in self.score_to_probability_func]):
            try:
                self.shift_lookup_table = read_csv(LOOKUP_TABLE_PATH, index_col=0)
            except Exception:
                logging.warn(
                    "Failed to load lookup table for a prespecified score to probability function. "
                    f"It is possible /data/{LOOKUP_TABLE_PATH}.csv is missing, in the wrong location, or corrupted. "
                    "Try rerunning /amputation/scripts.py to regenerate the lookup table."
                )
                self.shift_lookup_table = None

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
                patterns_with_missing_weights & (self.mechanisms == "MCAR")
            ] = np.zeros(shape=self.num_features)

            missing_mar_mask = patterns_with_missing_weights & (
                self.mechanisms == "MAR"
            )
            self.weights[missing_mar_mask] = self.observed_var_indicator[
                missing_mar_mask
            ]

            missing_mnar_mask = patterns_with_missing_weights & (
                self.mechanisms == "MNAR"
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
        assert (self.weights[self.mechanisms == "MCAR"] == 0).all(
            axis=None
        ), "Patterns with MCAR should have weights of all 0's."
        assert (
            not (self.weights[self.mechanisms != "MCAR"] == 0).all(axis=1).any()
        ), "Indicated weights of all 0's for a pattern that's not MCAR."
        assert all(
            [
                "weights" in self.patterns[pattern_idx]
                for pattern_idx in np.argwhere(self.mechanisms == "MAR+MNAR")[:, 0]
            ]
        ), "Failed to specify custom weights array for MAR+MNAR pattern."

        #####################################
        #     SCORE TO PROBABILITY FUNC     #
        #####################################
        assert (
            len(self.score_to_probability_func) == self.num_patterns
        ), "Score to probability functions must have an entry per pattern."
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
        self.num_features = X.shape[1]
        self.num_samples = X.shape[0]

        ##################
        #    PATTERNS    #
        ##################
        if self.patterns is None or len(self.patterns) == 0:
            logging.info("No patterns passed, setting default pattern.")
            self.patterns = self._get_default_pattern(self.num_features)
        assert isinstance(self.patterns, list) or isinstance(
            self.patterns, np.ndarray
        ), "Patterns should be a list of dictionaries."
        freq_keys = ["freq" in pattern for pattern in self.patterns]
        assert all(freq_keys) or not any(freq_keys), (
            "Either specify a freq for all patterns or specify none "
            "for equal frequency (1/k) for all patterns."
        )

        # check each dict has the required entries (via superset check)
        required_keys = {
            "incomplete_vars",
        }
        optional_keys = {"weights", "mechanism", "score_to_probability_func", "freq"}
        assert all(
            [set(pattern.keys()).issuperset(required_keys) for pattern in self.patterns]
        ), (
            "Patterns is malformed. "
            f"Each dict in the list must contain at least these keys: {required_keys}. "
            f"The {optional_keys} entries are optional."
        )
        # The dictionary form checks happens later
        assert all(
            [
                len(pattern["weights"]) == self.num_features
                for pattern in self.patterns
                if "weights" in pattern
                and (
                    isinstance(pattern["weights"], List)
                    or isinstance(pattern["weights"], np.ndarray)
                )
            ]
        ), "List of weights should be defined for every variable for every pattern."

        # bookkeeping vars for readability
        self.num_patterns = len(self.patterns)
        self.colname_to_idx = (
            {colname: idx for idx, colname in enumerate(X.columns)}
            if isinstance(X, DataFrame)
            else None
        )

        # converts patterms to matrix form for easy interal processing
        self._pattern_dict_to_matrix_form()

        self._load_shift_lookup_table()

        # defaults for the rest of the args (depends on patterns being initialized)
        self._set_defaults()
        self._validate_args()

        # vars involved in amputation have scores computed and need to be
        #   complete and numeric
        # A var (column) is involved if for any pattern (row) it has a weight.
        # We don't care about numeric restraint for MCAR
        self.vars_involved_in_ampute = (
            self.weights[self.mechanisms != "MCAR"] != 0
        ).any(axis=0)

        ##################
        #      DATA      #
        ##################
        assert X.shape[1] > 1, "Dataset passed must contain at least two columns."
        # enforce numpy just for checking
        X_check = X.values if isinstance(X, DataFrame) else X
        assert not isnull(
            X_check[:, self.vars_involved_in_ampute]
        ).any(), "Features involved in amputation must be complete, but contains NaNs."
        if not is_numeric(X_check[:, self.vars_involved_in_ampute]):
            logging.warn(
                "Features involved in amputation found to be non-numeric."
                " They will be forced to numeric upon calculating sum scores."
            )

        return X

    def fit_transform(self, X: Matrix) -> Matrix:
        """Fits amputer on complete data X and returns the incomplete data X

        Parameters
        ----------
        X : Matrix
            Matrix of shape `(n_samples, m_features)`
            Complete input data, where "n_samples" is the number of samples and
            "m_features" is the number of features.

        Returns
        -------
        X_incomplete : Matrix
            Matrix of shape `(n_samples, m_features)`.
            Incomplete dataset masked according to parameters.
        """

        # sets defaults, adjusts vars, and runs checks
        X = self._validate_input(X)

        # split complete_data in groups
        # the number of groups is defined by the number of patterns
        X_incomplete = X.copy()
        X_indices = np.arange(self.num_samples)
        # set seed for choice, if None it will be random.
        np.random.seed(self.seed)
        assigned_group_number = np.random.choice(
            a=self.num_patterns, size=self.num_samples, p=self.freqs
        )

        # start a loop over each pattern
        for pattern_idx in range(self.num_patterns):
            # assign cases to the group
            group_indices = X_indices[assigned_group_number == pattern_idx]
            pattern = np.squeeze(
                np.asarray(self.observed_var_indicator[pattern_idx, :])
            )
            data_group = (
                X[group_indices] if isinstance(X, np.ndarray) else X.iloc[group_indices]
            )
            # calculate weighted sum scores for each sample in the group
            wss = self._calculate_sumscores(data_group, pattern_idx)
            # define candidate probabilities in group
            probs = self._choose_probabilities(wss, pattern_idx)
            # apply probabilities and choose cases
            # set seed for random binomial
            np.random.seed(self.seed)
            chosen_candidates = np.random.binomial(
                n=1, size=data_group.shape[0], p=probs
            )
            # apply missing data pattern
            chosen_indices = group_indices[chosen_candidates == 1]
            if isinstance(X_incomplete, np.ndarray):
                X_incomplete[chosen_indices] = np.where(
                    pattern == 0, np.nan, X_incomplete[chosen_indices]
                )
            else:
                X_incomplete.iloc[chosen_indices, pattern == 0] = np.nan

        return X_incomplete
