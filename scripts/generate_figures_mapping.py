import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


from pyampute.ampute import MultivariateAmputation
from pyampute.exploration.md_patterns import mdPatterns


def generate_figures_mapping():

    nhanes2 = pd.read_csv("data/nhanes2.csv")
    mdp = mdPatterns()
    # set show_plot to True
    patterns = mdp.get_patterns(nhanes2, show_plot=False)

    mean = [5, 5, 5, 5]
    cor = 0.5
    cov = [
        [1, cor, cor, cor],
        [cor, 1, cor, cor,],
        [cor, cor, 1, cor],
        [cor, cor, cor, 1],
    ]
    n = 1000
    rng = np.random.default_rng()
    compl_dataset = rng.multivariate_normal(mean, cov, n)

    ma = MultivariateAmputation(
        patterns=[
            {"incomplete_vars": [3], "weights": [0, 4, 1, 0]},
            {"incomplete_vars": [2]},
            {"incomplete_vars": [1, 2], "mechanism": "MNAR"},
            {
                "incomplete_vars": [1, 2, 3],
                "weights": {0: -2, 3: 1},
                "mechanism": "MAR+MNAR",
            },
        ]
    )
    incompl_dataset = ma.fit_transform(compl_dataset)

    std_data = stats.zscore(compl_dataset)
    is_incomplete = np.where(np.isnan(incompl_dataset), "incompl", "compl")

    df0 = pd.DataFrame(
        dict(
            x=std_data[ma.assigned_group_number == 0, 1],
            y=ma.wss_per_pattern[0],
            label=is_incomplete[ma.assigned_group_number == 0, 3],
        )
    )

    df3 = pd.DataFrame(
        dict(
            x=std_data[ma.assigned_group_number == 3, 0],
            y=ma.wss_per_pattern[3],
            label=is_incomplete[ma.assigned_group_number == 3, 1],
        )
    )

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    for name, group in df0.groupby("label"):
        ax[0].plot(group.x, group.y, marker="o", linestyle="", ms=5, label=name)
    ax[0].legend()
    ax[0].set_xlabel("hyp")
    ax[0].set_title("wss pattern 1")

    for name, group in df3.groupby("label"):
        ax[1].plot(group.x, group.y, marker="o", linestyle="", ms=5, label=name)
    ax[1].legend()
    ax[1].set_xlabel("age")
    ax[1].set_title("wss pattern 4")

    fig.tight_layout()
    plt.savefig("docs/source/figures/wss_plots.png", dpi=600)


if __name__ == "__main__":
    generate_figures_mapping()
