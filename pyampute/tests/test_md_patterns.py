# test_md_patterns.py

import numpy as np
import pandas as pd

import pyampute.exploration.md_patterns as mp

if __name__ == "__main__":
    data_mar = pd.read_table("data/missingdata.csv", sep="\t")
    nhanes2 = pd.read_csv("data/nhanes2.csv")

    mypat = mp.mdPatterns()
    mdpatterns = mypat.get_patterns(data_mar)

    mypat = mp.mdPatterns()
    mdpatterns = mypat.get_patterns(nhanes2)

    X = np.random.randn(100, 3)
    mask1 = np.random.binomial(n=1, size=X.shape[0], p=0.5)
    print(mask1)
    mask2 = np.random.binomial(n=1, size=X.shape[0], p=0.5)
    X[mask1 == 1, 0] = np.nan
    X[mask2 == 1, 1] = np.nan
    print(
        X[1:10,]
    )

    mypat = mp.mdPatterns()
    mdpatterns = mypat.get_patterns(X)
