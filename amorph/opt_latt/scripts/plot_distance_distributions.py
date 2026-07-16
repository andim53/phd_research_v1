# Plot Distance Distributions
import matplotlib.pyplot as plt
import numpy as np

def plot_distance_distributions(
    distance_datasets,
    labels=None,
    title=None,
    xlabel="Distance (Å)",
    ylabel="Count",
    ref_dist=None,
    save_path=None,
    is_legend=False,
    rcParams = {
        'font.size': 15,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': 'black',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.grid': False,
        'legend.frameon': False
    },
    figsize=(4, 3),
    bins=10,
    alpha=1,
    edgecolor='black',
    color = 'black',
):
    """
    Plots distributions from multiple sets of distances. Each dataset is combined, optionally centered by a reference mean, then displayed as a single histogram.

    Parameters:
        distance_datasets (list of array-like): Each inner list or array contains distance values.
        labels (list of str): Labels corresponding to each dataset in distance_datasets.
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for the x-axis. Default is "Distance (Å)".
        ylabel (str, optional): Label for the y-axis. Default is "Count".
        ref_dist (array-like, optional): Reference distances used to center the data (mean of ref_dist is subtracted).
        save_path (str, optional): File path to save the plot. If None, the plot is only shown.
        is_legend (bool, optional): If True, displays a legend.
    """
    # === Style Configuration ===
    plt.rcParams.update(rcParams)

    # === Initialize Figure ===
    plt.figure(figsize=figsize)

    # === Plot Each Dataset ===
    for i, distances in enumerate(distance_datasets):
        all_distance = []
        for i, distances in enumerate(distance_datasets):
            all_distance.extend(distances)

        # If reference provided, shift distances by mean(ref_dist)
        if ref_dist is not None:
            ave_dist = np.mean(ref_dist)
            all_distance = [i - ave_dist for i in all_distance]

        plt.hist(
            all_distance,
            bins=bins,
            alpha=alpha,
            edgecolor=edgecolor,
            color=color,
        )

    # === Axis Labels and Title ===
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    # plt.grid(axis='y', alpha=0.3)

    # === Legend and Save ===
    if is_legend and labels:
        plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()
