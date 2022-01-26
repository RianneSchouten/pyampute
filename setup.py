#!/usr/bin/env python
from distutils.core import setup

setup(
    name="pyampute",
    version="0.0.1",
    description="Amputation suite based on the R MICE package.",
    author=["Rianne Schouten", "Davina Zamanzadeh", "Prabhant Singh"],
    author_email=["r.m.schouten@tue.nl", "davzaman@gmail.com", "p.singh@tue.nl"],
    packages=["pyampute"],
    install_requires=[
        # "rich",  # nice stack traces + printing
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
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
