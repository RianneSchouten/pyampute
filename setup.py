#!/usr/bin/env python
from distutils.core import setup

setup(
    name="pymice",
    version="0.0.1",
    description="Imputation suite based on R MICE package.",
    author=["Rianne Schouten", "Davina Zamanzadeh"],
    author_email=["riannemargarethaschouten", "davzaman@gmail.com"],
    packages=["pymice"],
    install_requires=[
        # "rich",  # nice stack traces + printing
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "",
    ],
    extras_require={
        "docs": ["sphinx", "pydata-sphinx-theme", "sphinx-autodoc-typehints"]
    },
)