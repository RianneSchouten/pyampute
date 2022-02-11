#!/usr/bin/env python
from distutils.core import setup
import setuptools

setup(
    name="pyampute",
    version="0.0.2",
    description="Transformer for generating multivariate missingness in complete datasets",
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
)
