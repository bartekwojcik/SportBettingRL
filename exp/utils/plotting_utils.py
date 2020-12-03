from typing import List

import matplotlib.pyplot as plt
from typing import List, Dict, Any
import os


def plot_error_bar(names: List[str], means: List[float], stds: List[float]) -> plt.Axes:
    """
    Simple box plot for comparing different algorithms

    :param names: list of strings
    :param means: list of floats
    :param stds: list od standard deviations
    :return:
    """


    fig, ax = plt.subplots()
    ax.errorbar(range(len(names)), means, yerr=stds, barsabove=True, fmt="rD")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=18)

    ax.set_title(f"models results (flat caps means std=0)")
    ax.set_ylabel("mean reward with standard deviation")
    ax.set_xlabel("Algorithm name and iteration number")

    return ax


def plot_results_error_bar(
    results: List[Dict[str, Any]], directory_to_save: str, file_name: str
):
    """
    Saves boxplot to file

    :param results:
    :param directory_to_save:
    :param file_name:
    :return:
    """

    names = [row["name"] for row in results]
    means = [row["mean"] for row in results]
    stds = [row["std"] for row in results]

    axis = plot_error_bar(names=names, means=means, stds=stds)
    f_path = os.path.join(directory_to_save, file_name)
    axis.get_figure().savefig(f_path)
