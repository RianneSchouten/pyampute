#!/usr/bin/env python
from distutils.core import setup
import setuptools

setup(
    name="pyampute",
    version="0.0.2",
    description="Transformer for generating multivariate missingness in complete datasets",
    long_description="Amputation is the opposite of imputation; it is the creation of a missing data mask for complete datasets. Amputation is useful for evaluating the effect of missing values on the outcome of a statistical or machine learning model. ``pyampute`` is the first open-source Python library for data amputation. Our package is compatible with the scikit-learn-style fit and transform paradigm, which allows for seamless integration of amputation in a larger, more complex data processing pipeline.",
    author="Rianne Schouten,Davina Zamanzadeh,Prabhant Singh",
    author_email="r.m.schouten@tue.nl,davzaman@gmail.com,p.singh@tue.nl",
    packages=setuptools.find_packages(
        include=["pyampute.*", "pyampute"],
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"],
    ),
    install_requires=[
        # "rich",  # nice stack traces + printing
        "pandas",
        "numpy>=1.19.0",
        "scipy",
        "matplotlib>=3.4.0",
        "scikit-learn",
    ],
    extras_require={
        "docs": [
            "sphinx",
            "pydata-sphinx-theme",
            "sphinx-autodoc-typehints",
            "sphinx-gallery",
        ]
    },
    project_urls={
        "Documentation": "https://rianneschouten.github.io/pyampute/build/html/index.html",
        "Source Code": "https://github.com/RianneSchouten/pyampute",
    },
    license="BSD",
)
