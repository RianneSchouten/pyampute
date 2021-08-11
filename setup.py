#!/usr/bin/env python
from distutils.core import setup

setup(
    name="pymice",
    version="1.0",
    description="Imputation suite based on R MICE package.",
    author=["Rianne Schouten", "Davina Zamanzadeh"],
    author_email=["riannemargarethaschouten", "davzaman@gmail.com"],
    packages=["pymice"],
    requires=[
        # "rich",  # nice stack traces + printing
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
    ],
)
