pymice: mice in Python
======================

.. role:: pyth(code)
  :language: python

The aim of library :pyth:`pymice` is to offer the large collection of missing data methods to the Python community.

The intent is to create three packages of functions:

1. missing data exploration: offer functions to inspect missing data and its characteristics
2. multivariate amputation: implement the methodology of mice::ampute in Python
3. multiple imputation: implement the methodology of mice in Python

At the moment, the package contains class :pyth:`McarTests`. This class of functions consists of two functions to inspect whether the nonresponse has a MCAR missingness mechanism. Little’s MCAR test is implemented in :pyth:`mcar_test` and for each pair of variables, t-tests can be performed with function :pyth:`mcar_t_tests`.

Obviously, a lot of development has still to be done.

My contact details are here_

.. _here: https://rianneschouten.github.io/#contact