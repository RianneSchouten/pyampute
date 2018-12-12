"""Transformer for generating multivariate missingness in complete datasets"""
# Author: Rianne Schouten <riannemargarethaschouten@gmail.com>

import numpy as np
from sklearn.base import TransformerMixin
from scipy import stats


class MultivariateAmputation(TransformerMixin):
    """Generating multivariate missingness patterns in complete datasets

    Some short explanation.

    Parameters
    ----------
    complete_data : matrix with shape (A, B)

    prop : float

    patterns : matrix with shape (C, B)

    freq : array of length B

    weights : matrix with shape (C, B)

    std : boolean

    mechanism: array of length B

    logit_type : array of length B

    lower_range : float

    upper_range : float

    max_dif_with_target : float

    max_iter : integer

    Attributes
    ----------
    incomplete_data : 

    Notes
    -----
    Something on difference ampute in R and Python

    References
    ----------
    .. [1] Rianne Margaretha Schouten, Peter Lugtig & Gerko Vink (2018). 
    Generating missing values for simulation purposes: A multivariate amputation procedure. 
    Journal of Statistical Computation and Simulation, DOI: 10.1080/00949655.2018.1491577
    """


    def __init__(self,
                 prop=0.5, 
                 patterns=None, 
                 freqs=None, 
                 weights=None, 
                 std=True, 
                 mechanisms=None,
                 types=None,
                 lower_range=-3,
                 upper_range=3,
                 max_dif_with_target=0.001,
                 max_iter=100):

        self.prop = prop
        self.patterns = patterns
        self.freqs = freqs
        self.weights = weights
        self.std = std
        self.mechanisms = mechanisms
        self.types = types
        self.lower_range = lower_range
        self.upper_range = upper_range
        self.max_dif_with_target = max_dif_with_target
        self.max_iter = max_iter


    def _binary_search(self, wss_standardized, i): 
  
        b = 0
        counter = 0
        lower_range = self.lower_range
        upper_range = self.upper_range

        # start binary search with a maximum amount of tries of max_iter
        while counter < self.max_iter:
            counter += 1
          
            # in every iteration, the new b is the mid of the lower and upper range
            # the lower and upper range are updated at the end of each iteration
            b = lower_range + (upper_range - lower_range)/2
            if counter == self.max_iter: break 
     
            # calculate the expected missingness proportion
            # depends on the logit type, the sum scores and b
            x = np.zeros(len(wss_standardized))
            if self.types[i] == 'RIGHT': 
                x = wss_standardized + b
            elif self.types[i] == 'LEFT':
                x = -wss_standardized + b
            elif self.types[i] == 'MID':
                x = -np.absolute(wss_standardized) + 0.75 + b
            elif self.types[i] == 'TAIL':
                x = np.absolute(wss_standardized) - 0.75 + b
            probs = 1 / (1 + np.exp(-x))
            current_prop = np.sum(probs) / len(x)

            # if the expected proportion is close to the target, break
            # the maximum difference can be specified
            # if max_dif_with_target is 0.001, the proportion differs with max 0.1%
            if np.absolute(current_prop - self.prop) < self.max_dif_with_target: break

            # if we have not reached the desired proportion
            # we adjust either the upper or lower range
            # this way of adjusting works for self.types[i] = 'RIGHT'
            # need to check for the other types
            # in the next iteration, a new b is then calculated and used
            if (current_prop - self.prop) > 0: 
               upper_range = b
            else:
               lower_range = b 

        return b    


    def _choose_probabilities(self, wss, i):

        # when wss contains merely zeroos, the mechanism is MCAR
        # then each case has an equal probability of becoming missing
        if  np.all(wss == 0):
            probs = np.repeat(self.freqs[i], len(wss))
        # else we calculate the probabilities based on the wss
        else:
            # standardize wss
            wss_standardized = stats.zscore(wss)
            # calculate the size of b for the desired missingness proportion
            b = self._binary_search(wss_standardized, i)
            # apply the right logistic function
            x = np.zeros(len(wss))
            if self.types[i] == 'RIGHT':
                 x = wss_standardized + b
            elif self.types[i] == 'LEFT':
                x = -wss_standardized + b
            elif self.types[i] == 'MID':
                x = -np.absolute(wss_standardized) + 0.75 + b
            elif self.types[i] == 'TAIL':
                x = np.absolute(wss_standardized) - 0.75 + b
            # calculate probability to be missing 
            probs_matrix = 1 / (1 + np.exp(-x))
            probs = np.squeeze(np.asarray(probs_matrix))
        
        return probs


    def _calculate_sumscores(self, data_group, i):

        # transform categorical data to numerical data
        # standardize data or not
        if self.std:
            data_group = stats.zscore(data_group)
        
        # calculate sum scores
        # in case of MCAR, weights[i, ] contains merely zeroos and wss are merely zeroos
        # in case of MAR, MNAR, the mechanisms is determined by the weights
        wss = np.dot(data_group, self.weights[i, ].T)

        return wss


    def _validate_input(self, X):

        # default patterns is missingness on each variable
        if self.patterns is None:
            self.patterns = 1 - np.identity(n=X.shape[1])
        #else: 
            #check if self.patterns.shape[1] = X.shape[1]
            #check if self.patterns does not contain values other than 0 and 1
            #check if there are no patterns with merely 0 and merely 1

        # default freqs is every pattern occurs with the same frequency
        if self.freqs is None:
            self.freqs = np.repeat(1 / self.patterns.shape[0], self.patterns.shape[0])
        #else: 
            #check if freq has length equal to self.patterns.shape[0]  
            #check if freq does not have values < 0 or > 1

        # default mechanisms is MAR mechanism for each pattern 
        if self.mechanisms is None:
            self.mechanisms = np.repeat('MAR', len(self.freqs))
        elif len(self.mechanisms) == 1:
            self.mechanisms = np.repeat(self.mechanisms[0], len(self.freqs))
        #else: 
            #check if mechanism has length equal to self.patterns.shape[0]     
            #check if mechanism has no other values than 'MAR', 'MNAR' or MCAR   

        # default in case of MAR: all observed variables have weight 1
        # default in case of MNAR: all non observed variables have weight 1
        # with MCAR: all variables should have weight 0
        if self.weights is None:
            self.weights = np.zeros(shape=self.patterns.shape)
            self.weights[self.mechanisms == 'MAR', ] = self.patterns[self.mechanisms == 'MAR', ]
            self.weights[self.mechanisms == 'MNAR', ] = 1 - self.patterns[self.mechanisms == 'MNAR', ]
        #else:
            #check if self.weights.shape equals self.patterns.shape
            #check if the patterns with MCAR contain merely zeroos

        if self.types is None: 
            self.types = np.repeat('RIGHT', len(self.mechanisms))
        #else:
            # check if len equals len mechanisms and len freqs
            # check if types contains no other words then right, left, mid and tail

        return X

        
    def fit_transform(self, X):
        """Fits amputer on complete data X and returns the incomplete data X
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Complete input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        Returns
        -------
        X_incomplete : array-like, shape (n_samples, n_features)
        """

        # some check functions
        # including specification of defaults
        X = self._validate_input(X)

        # split complete_data in groups
        # the number of groups is defined by the number of patterns
        # we know the number of patterns by the number of rows of self.patterns
        X_incomplete = np.zeros(X.shape)
        X_indices = np.arange(X.shape[0])
        assigned_group_number = np.random.choice(a=self.patterns.shape[0],
                                                 size=X.shape[0], p=self.freqs)
        
        # start a loop over each pattern
        for i in range(self.patterns.shape[0]):
            # assign cases to the group
            group_indices = X_indices[assigned_group_number == i]
            pattern = np.squeeze(np.asarray(self.patterns[i, ]))
            data_group = X[group_indices]
            # calculate weighted sum scores for each group
            wss = self._calculate_sumscores(data_group, i)
            # define candidate probabilities in group
            probs = self._choose_probabilities(wss, i)
            # apply probabilities and choose cases
            chosen_candidates = np.random.binomial(n=1,
                                                   size=data_group.shape[0],
                                                   p=probs)
            # apply missing data pattern
            chosen_indices = group_indices[chosen_candidates==1]
            X_incomplete[chosen_indices, pattern == 0] = np.nan

        return X_incomplete


