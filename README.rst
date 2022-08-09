pyampute
========
|made-with-python| |code-coverage|

.. |made-with-python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/

.. |code-coverage| image:: https://img.shields.io/codecov/c/github/RianneSchouten/pyampute
   :target: https://app.codecov.io/gh/RianneSchouten/pyampute/

.. role:: pyth(code)
  :language: python

A Python library for generating missing values in complete datasets (i.e. amputation) and exploration of incomplete datasets. 

Check out the `documentation and find examples`_!

.. _`documentation and find examples`: https://rianneschouten.github.io/pyampute/build/html/index.html

Features
--------

Amputation is the opposite of imputation: the generation of missing values in complete datasets. This is useful for evaluating the effect of missing values in your model, mostly in experimental settings, but also as a preprocessing step in developing models.

Our `MultivariateAmputation`_ class is compatible with the scikit-learn-style ``fit`` and ``transform`` paradigm and can be used in a scikit-learn ``Pipeline``.

The underlying methodology has been proposed by `Schouten, Lugtig and Vink (2018)`_ and has been implemented in an R-function as well: `mice::ampute`_. Compared to ``ampute``, ``pyampute``'s parameters are easier to specify and allow for more variation. See `this blogpost`_ to learn more.

.. _`Schouten, Lugtig and Vink (2018)`: https://www.tandfonline.com/doi/full/10.1080/00949655.2018.1491577
.. _`mice::ampute`: https://rianneschouten.github.io/mice_ampute/vignette/ampute.html
.. _`this blogpost`: https://rianneschouten.github.io/pyampute/build/html/mapping.html
.. _`MultivariateAmputation`: https://rianneschouten.github.io/pyampute/build/html/pyampute.ampute.html

.. code-block:: python

   import numpy as np
   from pyampute.ampute import MultivariateAmputation
   n = 1000
   m = 10
   X_compl = np.random.randn(n,m)
   ma = MultivariateAmputation()
   X_incompl = ma.fit_transform(X_compl)

Among others, we also provide an `mdPatterns`_ class, which displays missing data patterns in incomplete datasets.

.. code-block:: python

   from pyampute.exploration.md_patterns import mdPatterns
   mdp = mdPatterns()
   patterns = mdp.get_patterns(X_incompl)

.. _`mdPatterns`: https://rianneschouten.github.io/pyampute/build/html/pyampute.exploration.html

Installation
------------
Python Package Index (PyPI)
***************************

::

   pip install pyampute

From source
***********

::

   git clone https://github.com/RianneSchouten/pyampute.git
   pip install ./pyampute

License
-------

BSD 3-Clause License

Citation
--------

.. code-block:: bibtex

   @misc{schouten_rianne_m_2022_6946887,
   author       = {Schouten, Rianne M and
                  Zamanzadeh, Davina and
                  Singh, Prabhant},
   title        = {pyampute: a Python library for data amputation},
   month        = aug,
   year         = 2022,
   publisher    = {Zenodo},
   doi          = {10.25080/majora-212e5952-03e},
   url          = {https://doi.org/10.25080/majora-212e5952-03e}
   }

   @article{Schouten2018,
   title={Generating missing values for simulation purposes: {A} multivariate amputation procedure},
   author={Schouten, Rianne M. and Lugtig, Peter and Vink, Gerko},
   journal={Journal of Statistical Computation and Simulation},
   volume={88},
   number={15},
   pages={2909--2930},
   year={2018}
   }

Watch our `SciPy'22 presentation here`_.

.. _`SciPy'22 presentation here`: https://www.youtube.com/watch?v=jMEzKFV-ilc&list=PLYx7XA2nY5GcBWLGTzhJ1vxGtHIcyHrRr&index=3.

Contact details
---------------

For questions, comments and if you would like to contribute, please do not hesitate to contact us. You can `find our contact details here`_.

Cheers,

.. _`find our contact details here`: https://rianneschouten.github.io/pyampute/build/html/about.html


