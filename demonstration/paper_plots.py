import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_joinability_scores():
    sns.set(font_scale=1.5)

    scores = np.array([2914, 969, 969, 969, 969, 969, 969, 969, 969, 969, 969,
                       969, 969, 969, 969, 969, 969, 352, 352, 352], dtype=int)

    scores = scores / int(pd.read_csv("../datasets/movie.csv").shape[0])

    plot_data = pd.DataFrame([], columns=["Table Rank", "Joinability Score"])
    plot_data["Table Rank"] = np.arange(1, len(scores) + 1)
    plot_data["Joinability Score"] = scores

    g = sns.catplot(data=plot_data, x="Table Rank", y="Joinability Score")
    g.fig.set_size_inches(9.5, 3)

    plt.tight_layout()

    plt.savefig("../fig/joinability_scores_v3.png", dpi=500)
    plt.show()


def plot_correlation_heatmap():
    sns.set(font_scale=1.5)

    idx_y= ['Movie Title', 'Director Name', 'IMDB Score', '#Voted Users', 'Duration', '#Reviewers']
    idx_x = ['Movie\nTitle', 'Director\nName', 'IMDB\nScore', '#Voted\nUsers', 'Duration', '#Reviewers']

    corr = pd.read_csv("../temp_data/correlation.csv", index_col=0)

    corr.columns = idx_x
    corr.index = idx_y

    plt.figure(figsize=(9.5, 2.7), constrained_layout=True)
    heatmap = sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap=sns.color_palette("vlag", as_cmap=True)
    )

    plt.xticks(rotation=0)
    plt.savefig("../fig/joinability_scores_v3.png", dpi=500)
    plt.show()


if __name__ == "__main__":
    #plot_joinability_scores()
    plot_correlation_heatmap()





