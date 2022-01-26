A mapping from R-function ``ampute`` to ``pyampute``
====================================================

Multivariate amputation has been proposed by [Schouten2018]_ and implemented in function ``ampute`` in statistical language ``R`` in package ``mice``. Since then, the amputation methodology has been used and cited by many. 

With ``pyampute``, we now provide the same methodology for Python users, and more. Compared to ``ampute``, the input arguments of ``pyampute`` are more intuitive and faster to specify. A quick explanation of ``pyampute``'s arguments can be found in the `documentation`_. 

For the R-function, a detailed explanation of how the multivariate amputation methodology relates to the input arguments is provided in a `vignette`_. This blogpost therefore has two purposes: 

1. For R users, we provide a mapping from the input arguments of ``ampute`` to those of ``pyampute``. 
2. For Python users, we further explain how the input arguments of ``pyampute`` can be used to generate any desired form of missing data. 

.. _vignette: https://rianneschouten.github.io/mice_ampute/vignette/ampute.html
.. _mice: https://github.com/amices/mice
.. _documentation: https://rianneschouten.github.io/pymice/build/html/pymice.amputation.html

The fundament: patterns
-----------------------

Key in multivariate amputation is the specification of missing data patterns. A missing data pattern is a combination of missing values on certain variables and observed values on the other variables. Knowing the missing data patterns in an incomplete dataset can be very helpful for understanding possible explanations for the occurrence of those missing values. 

In ``pyampute``, an overview of missing data patterns in an incomplete dataset can be obtained with class ``mdPatterns``. For instance, the `nhanes2`_ dataset has 4 missing data patterns:

.. code-block:: python

    import pyampute.md_patterns as mp 
    mypat = mp.mdPatterns()
    mdpatterns = mypat.get_patterns(nhanes2)

.. figure:: figures/mdpatterns_nhanes2.png
    :width: 200px
    :align: center
    :alt: alternate text
    :figclass: align-center

    Missing data patterns of incomplete dataset nhanes2

Here, blue and red correspond to observed and missing values respectively. The numbers on the left indicate the number of rows in the dataset that follow a specific missing data pattern. For instance, there are 3 rows with observed values on variables age, hyp and bmi and a missing value on variable chl. In this dataset, there are no rows with observed values on age and hyp and missing values on bmi and chl. That combination does not exist. 

All aspects of multivariate amputation are connected to the missingness patterns. Per pattern, one decides the missing data mechanism, the frequency, the weights of the variables that guide the amputation and the missingness type. Considering this, the input arguments of ``ampute`` and ``pyampute`` differ as follows.

.. topic:: Difference between ``ampute`` and ``pyampute``

    In ``ampute``, the patterns, mechanisms, weights, frequencies and types are defined in distinct arguments, and they should all describe the same number of patterns. For instance, the first row in the patterns matrix should correspond to the first row in the weights matrix, and to the first value in the frequency vector, etcetera.

    In ``pyampute``, one specifies one dictionary per pattern. That dictionary contains information about the pattern and the corresponding mechanism, weights, frequency and type. One can then easily add or remove patterns.

Let's discuss in more detail.

.. _nhanes2: https://github.com/RianneSchouten/pymice/tree/master/data

The patterns matrix in ``ampute``
*********************************

In ``ampute``, patterns are specified with a binary matrix of size :math:`k` by :math:`m` where :math:`k` indicates the number of patterns and :math:`m` indicates the number of variables. A cell in the patterns matrix equals ``0`` if in that patterns variable :math:`m` should be amputed and ``1`` otherwise.

The patterns matrix of `nhanes2`_ would look as follows:

.. math::

    patterns = \begin{bmatrix}
            1 & 1 & 1 & 0 \\
            1 & 1 & 0 & 0 \\
            1 & 0 & 0 & 1 \\
            1 & 0 & 0 & 0
        \end{bmatrix}

Without further specification, by default ``ampute`` generates the number of patterns equal to the number of variables in the dataset. Then, in every pattern, just one variable is amputed. For instance, if we would ampute a complete version of nhanes2, the default patterns matrix in ``ampute`` would be:

.. math::

    default = \begin{bmatrix}
            0 & 1 & 1 & 1 \\
            1 & 0 & 1 & 1 \\
            1 & 1 & 0 & 1 \\
            1 & 1 & 1 & 0
        \end{bmatrix}

.. _nhanes2: https://github.com/RianneSchouten/pymice/tree/master/data

Specifying patterns in ``pyampute``
***********************************

As noted before, in ``pyampute`` we make all specifications per pattern. In case of multiple patterns, the user specifies a list of dictionaries. For `nhanes2`_, the input would be:

.. code-block:: python

    import pyampute.pyampute as ampute 
    ma = ampute.MultivariateAmputation(
        patterns = [
            {'incomplete_vars': [3]},
            {'incomplete_vars': [2,3]},
            {'incomplete_vars': [1,2]},
            {'incomplete_vars': [1,2,3]}
        ]
    )

This may seem cumbersome at first, but it will allow for easy modification of a single pattern why keeping the others intact. We will further discuss this when talking about mechanisms, weights, frequency and types. 

We have furthermore adapted the default. In ``pyampute``, by default we generate one missing data pattern with missing values on a random selection of 50% of the variables. We expect this default to be possible for many different datasets, whereas in ``ampute`` an error occurs if the number of variables is high compared to the desired (or default) missingness proportion.

Proportion and frequency
------------------------

Naturally, it is important to specify the proportion of missing values. In multivariate amputation, we control that proportion with two input arguments:

1. ``prop`` determines the proportion of incomplete data rows
2. ``freq`` determines how those incomplete rows are divided over the :math`k` patterns.

Proportion and frequency in ``ampute``
**************************************

In ``ampute``, ``prop`` is a float (or integer) and ``freq`` is a vector of floats of length :math`k`. Thus, ``prop = 0.3`` indicates that 30% of the rows should be amputed. Then, ``freq = c(0.1, 0.1, 0.2, 0.6)`` indicates that 10% of those incomplete rows should have pattern 1, 10% should have pattern 2, 20% should have pattern 3 and 60% should have pattern 4. In this way, you can define how the missing values are divided over your dataset.

Proportion and frequency in ``pyampute``
****************************************

In ``pyampute``, the definitions of proportion and frequency are very similar to those in ``ampute``. The only difference is that ``prop`` is a global input argument that is specified outside the patterns input, whereas ``freq`` is specified per pattern, and can therefore be integrated in the dictionaries. That looks as follows:

.. code-block:: python

    ma = ampute.MultivariateAmputation(
        patterns = [
            {'incomplete_vars': [3], 'freq': 0.1},
            {'incomplete_vars': [2,3], 'freq': 0.1},
            {'incomplete_vars': [1,2], 'freq': 0.2},
            {'incomplete_vars': [1,2,3], 'freq': 0.6}
        ],
        prop = 0.3
    )

It is important to realize that all incomplete rows should follow at least one of the patterns, and therefore the frequency values should sum to 1. 

Missing data mechanisms
-----------------------

In missing data methodology, we categorize missing data problems into three categories (cf. [Rubin1976]_):

1. Data is Missing Completely At Random (MCAR) if the probability of being incomplete is the same for every row in the dataset.
2. Data is Missing At Random (MAR) if the probability of being incomplete depends on the observed values.
3. Data is Missing Not At Random (MNAR) if the probability of being incomplete is unobserved. That could mean the probability depends on the missing values themselves, or it means the probability depends on a source outside the data. 

Which of the three missing data mechanisms apply (or can be assumed) greatly determines the effect of missing values on the outcome of any data analysis or model. For more information and examples, we gladly refer to [Schouten2021]_ or [VanBuuren2018]_.

Mechanisms in ``ampute``
************************

In ``ampute``, mechanisms are defined using a string or a vector of strings that equals the number of patterns. Thus, ``mech = "MAR"`` or ``mech = c("MAR", "MCAR", "MAR", "MNAR")`` (for four patterns).

Mechanisms in ``pyampute``
**************************

In ``pyampute``, we specify the mechanism with a string per pattern. Here, it is not necessary to define a mechanism for every pattern. If left open, the MAR default will be used. 

.. code-block:: python

    ma = ampute.MultivariateAmputation(
        patterns = [
            {'incomplete_vars': [3], 'mechanism': "MCAR"},
            {'incomplete_vars': [2,3]},
            {'incomplete_vars': [1,2], 'mechanism': "MNAR"},
            {'incomplete_vars': [1,2,3]}
        ]
    )

The missing data mechanisms are strongly related to the concept of *weighted sum scores*, which will be discussed in the next section. In case one chooses a MCAR, MAR or MNAR mechanism, these choices will be translated into a default set of weights. One may also want to determine a custom set of weights, which may also result in MAR or MNAR missingness. In that case, the mechanism does not have to explicitly be specified.

In ``pyampute``, we added an option of setting the mechanism to be a mixture of MAR and MNAR missingness. For that pattern, weights *have to* be provided. We will now discuss the exact meaning of these weights. 

Weighted sum scores: weights
----------------------------

In multivariate amputation, we not only control which variables are missing together (by means of a pattern), but also which variables determine the amputation together (in case of MAR and MNAR). For instance, with a MAR mechanism, one may want multiple variables to influence the amputation process, because that means that the observed information of all those variables need to be used for imputation, which can be challenging.

Whether or not a data row is amputed is determined by the combination of a weighted sum score and a probability distribution. For row :math:`i` in :math:`\{1,2,...,n\}`, the weighted sum score is calculated as follows:

.. math::

    wss_i = w_1 y_{i1} + w_2 y_{i2} + ... + w_m y_{1m},

where :math:`w_j` is the weight for variable :math:`y_j` with :math:`j` in :math:`\{1,2,...,m\}`. By setting certain weights to 0, we have full control over whether a mechanism is MAR, MNAR or a combination of both. 

By default, a MAR mechanism will translate into zero weights for the incomplete variables and weights of 1 for the complete variables. For MNAR, zero weights are given to the complete variables and 1s to the incomplete variables. Consequently, by default, all variables in the weighted sum score are equally weighted. 

Weights in ``ampute``
*********************

In ``ampute``, weights are specified in a matrix that has the same size as the patterns matrix; :math:`k` by :math:`m`. In other words, the weights for pattern 1 are specified in the first row of the weights matrix, the weights for the second pattern in the second row, etcetera. 

Weights in ``pyampute``
***********************

In ``pyampute``, weights are per pattern. That gives the flexibility to manually define weights for some patterns, but use the default settings for other patterns. Weights can be specified using an array or a dictionary.

For instance, consider the following specification:

.. code-block:: python

    ma = ampute.MultivariateAmputation(
        patterns = [
            {'incomplete_vars': [3], 'weights': [0,4,1,0]},
            {'incomplete_vars': [2,3]}},
            {'incomplete_vars': [1,2], 'mechanism': "MCAR"},
            {'incomplete_vars': [1,2,3], 'weights': {'2':-1, '3':1}}
        ]
    )

Here, for the first pattern, we specify weights using an array. Non-zero weights are given to the second and third variables; hyp and bmi. Since in pattern 1 both these variables are not amputed, this will lead to a MAR mechanism. However, it differs from a default MAR mechanism, which would be ``[1,1,1,0]``, in two ways:

1. Variable age is not weighted, even though it is a complete variable in pattern 1
2. The effect of variable hyp is 4 times as large as the effect of variable bmi. 

For the fourth pattern, we specify weights using a dictionary. Here, variables hyp and bmi are weighted. Both these variables are amputed, and therefore these weights correspond to a MNAR mechanism. Note that variable hyp has a negative effect on the weighted sum score; a higher value on hyp will result in a lower weighted sum score, whereas a higher value on bmi will result in a higher weighted sum score. In this way you can control which data rows should be amputed (and where the resulting bias exists).

Note that whether or not the effect of one variable is really higher than the effect of the other variable depends on the scale of these variables as well. Therefore, both ``ampute`` and ``pyampute`` by default standardize the dataset before calculating weighted sum scores. One can turn this off by ``std = False``.

Missingness types or score_to_probability_func
----------------------------------------------

At this point in the amputation process, all data rows will have been assigned a weighted sum score. Consequently, these scores are used to determine which cases should be amputed. We do that by means of a probability function that maps the weighted sum scores to a probability of being amputed. The multivariate amputation methodology knows four types of probability functions.

.. figure:: figures/types.pdf
    :width: 200px
    :align: center
    :alt: alternate text
    :figclass: align-center

    Four types of probability functions

First, weighted sum scores are standardized. Then, the probability function is shifted such that the desired ``prop`` will be reached. And then the probability function will be applied. 

Types in ``ampute``
*******************

Similar like ``mech`` and ``freq``, in ``ampute`` the missingness types are specified with a vector that should have length :math`k`: ``type = c("LEFT", "RIGHT", "MID", "TAIL")``. 

Score_to_probability_func in ``pyampute``
*****************************************

In ``pyampute``, we have renamed the input argument to make the meaning of the argument more clear. Similar as the specifications for weights and mechanism, a probability function can be chosen per pattern. 

In addition, ``pyampute`` allows for the specification of a custom probability function. This can be any function that maps values in the range :math`[-inf,inf] \rightarrow [0,1]`. In the example `Amputing with a Custom Probability Function`_, the working will be further demonstrated.

.. _Amputing with a Custom Probability Function: file:///C:/Users/20200059/Documents/Github/pyampute/docs/build/html/auto_examples/plot_custom_probability_function.html

Full specification for ``pyampute``
-----------------------------------

Altogether, we could do something as follows:

.. code-block:: python

    ma = ampute.MultivariateAmputation(
        prop = 0.4,
        patterns = [
            {'incomplete_vars': [3], 'freq': 0.1, 'mechanism': "MCAR", 'score_to_probability_func': "sigmoid-left"},
            {'incomplete_vars': [2,3]}, 'freq': 0.1, 'weights': [2,3], 'score_to_probability_func': "sigmoid-mid"},
            {'incomplete_vars': [1,2], 'freq': 0.2, 'mechanism': "MAR+MNAR", 'weights': [2,3]},
            {'incomplete_vars': [1,2,3], 'freq': 0.6, 'mechanism': "MCAR"}
        ],
        std = True
    )

Here a blogpost about a mapping.

One asterisk for *italics* and two for **bolding** and backticks for ``code samples``.

`A link is provided by backticks`_ and you can define it later.

.. _A link is provided by backticks: https://rianneschouten.github.io/pymice 

Later we will discuss something in the `This is a subheader`_

This is a subheader
-------------------

The default color coding is Python::

    mads <- MultivariateAmputation(patterns = )
    incompl_data <- mads.fit_transform(compl_data)

If you want to change the color coding for R code, you have to do:

.. code-block:: r

    mads <- ampute(compl_data, patterns = matrix(c(1,0,0,0,0,1),nrow=2,byrow=TRUE))
    incompl_data <- mads$data

Patterns and weights are specified in matrices. Consider a dataset with 3 variables; V1, V2 and V3.
Assume you want to create two missing data patterns. In the first, you create missingness in V1, in the second you create missingness in V1 and V2.
The patterns matrix will be:

.. tabularcolumns:: |c|c|r|

+------+------+------+
| V1   | V2   |  V3  |
+------+------+------+
| 0    |  1   | 1    |
+------+------+------+
| 0    |  0   | 1    |
+------+------+------+

Or with csv code;

.. csv-table:: a title
   :header: "V1", "V2", "V3"
   :widths: 1, 1, 2   
   :width: 30

   0, 1, 1
   0, 0, 1

If we denote a weight as :math:`w_{ij}`, a weighted sum score is then given as

.. math::

    n_{\mathrm{offset}} = \sum_{k=0}^{N-1} s_k n_k

And we may want to add a figure

.. figure:: figures/scheme.png
    :width: 600px
    :align: center
    :height: 300px
    :alt: alternate text
    :figclass: align-center

    figure are like images but with a caption

A citation is defined at the bottom of the page and referenced as [Schouten2018]_ and [Schouten2021]_

References
----------
.. [Schouten2018] Rianne M. Schouten, Peter Lugtig and Gerko Vink, etc. 
.. [Schouten2021] Rianne M. Schouten and Gerko Vink, etc. 
.. [Rubin1976] Reference to Rubin
.. [VanBuuren2018] Refernece to van Buuren