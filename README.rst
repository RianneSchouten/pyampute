pymice
======
|made-with-python| |code-coverage|

.. |made-with-python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/

.. |code-coverage| image:: https://img.shields.io/codecov/c/github/RianneSchouten/pyampute
   :target: https://app.codecov.io/gh/RianneSchouten/pyampute/


.. role:: pyth(code)
  :language: python

A Python library for generating missing values in complete datasets (i.e. amputation) and exploration of incomplete datasets. 

Check out the `documentation`_!

.. _documentation: https://rianneschouten.github.io/pyampute/build/html/index.html

amputation.ampute
-----------------

The MultivariateAmputation class is an implementation of the multivariate amputation methodology by `Schouten, Lugtig and Vink (2018)`_. It is designed as an sklearn TranformerMixin class to allow for easy integration with pipelines. 
.. _Schouten, Lugtig and Vink (2018): https://www.tandfonline.com/doi/full/10.1080/00949655.2018.1491577

Compared to the implementation in ``mice:ampute`` in **R**, ``pymice.amputation.ampute`` has a few extra functionalities:

1. The function's arguments are more intuitive. In this `blog post`_, we provide a mapping.
2. The method allows for custom probability functions, see this `example`_.
3. The function allows for non-numerical data features, as long as they are not used as observed data in MAR amputation.

.. _blog post: https://rianneschouten.github.io/pymice/build/html/index.html
.. _example: https://rianneschouten.github.io/pymice/build/html/index.html

exploration.md_patterns
----------------------

Extra exploration functions are available to explore incomplete datasets. 

The ``mdPatterns`` class is an implementation of ``mice:md.pattern`` in **R** and gives a quick overview of the missingness patterns::

.. code-block:: python

   from pymice.exploration.mdpattern import mdPatterns
   my_pat = mdPatterns(inc_data)
   my_pat.summary()
   my_pat.visualization()


installation
------------

via pip
```
pip install pyampute
```

