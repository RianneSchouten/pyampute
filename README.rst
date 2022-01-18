pymice
======

.. role:: pyth(code)
  :language: python

A Python library for generating missing values in complete datasets (i.e. amputation) and exploration of incomplete datasets. 

Check out the [documentation](https://rianneschouten.github.io/pymice/build/html/index.html)!

amputation.ampute
=================

The MultivariateAmputation class is an implementation of the multivariate amputation methodology by Schouten, Lugtig and Vink (2018). It is designed as an sklearn TranformerMixin class to allow for easy integration with pipelines. 

Compared to the implementation in `mice:ampute` in **R**, `pymice.amputation.ampute` has a few extra functionalities:

1. The function's arguments are more intuitive. In this [blog post](https://rianneschouten.github.io/pymice/build/html/index.html), we provide a mapping.
2. The method allows for custom probability functions, see this [example](https://rianneschouten.github.io/pymice/build/html/index.html).
3. The function allows for non-numerical data features, as long as they are not used as observed data in MAR amputation.

exploration.mdpatterns
======================

Extra exploration functions are available to explore incomplete datasets. 

With `mdPatterns`, a quick overview of missingness patterns can be created.

```
from pymice.exploration.mdpattern import mdPatterns
my_pat = mdPatterns(inc_data)
my_pat.summary()
my_pat.visualization()
```

installation
============
