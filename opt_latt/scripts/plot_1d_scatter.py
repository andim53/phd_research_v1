import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from typing import Optional, Sequence, Dict, Any


def plot_1d_scatter(
    data_1d: np.ndarray,
    data_idx: np.ndarray = None,
    rcParams: Dict[str, Any] = {
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': True,
        'axes.edgecolor': 'black',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.grid': False,
        'legend.frameon': False,
    },
    figsize: Sequence[float] = (4, 2),
    xlabel: str = r"$E_{rel}$(meV/atom)",
    xlim: Sequence[float] = (0, 0.1),
    xticks: np.ndarray = np.arange(0, 0.11, 0.01),
    show_labels: bool = True,
    save_path: Optional[str] = None,
    scale_up: float = 1000,
) -> None:
    """
    Create a 1D scatter visualization using short vertical markers.

    Args:
        data_1d: 1D array of numeric values.
        rcParams: Matplotlib parameters applied before rendering.
        figsize: Figure size.
        xlabel: Label for the x-axis.
        xlim: Plotting range for x-axis.
        xticks: Tick positions on x-axis.
        show_labels: If True, annotate points with indices.
        save_path: If provided, save the figure to this path.
        scale_up: Multiplicative factor for displayed tick values.
    """

    plt.rcParams.update(rcParams)

    fig, ax = plt.subplots(figsize=figsize)

    # draw vertical tick-like markers at values
    ax.vlines(data_1d, ymin=0, ymax=0.01, color='black', alpha=1)

    if show_labels:
        for i, val in zip(data_idx, data_1d):
            ax.text(val, 0.01, f"{i}", ha='center', va='bottom',
                    fontsize=10, rotation=0)

    # scale tick labels
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x * scale_up:.0f}")
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_ylabel("")

    ax.tick_params(axis='x', width=1.5, length=5)
    ax.set_xlabel(xlabel)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xticks(xticks)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.close()


# from scripts.plot_1d_scatter import plot_1d_scatter
# import pandas as pd

# energy_limit = (0, 0.1) # in eV
# csv_i = f"{dir_out}/data_0.csv"

# df = pd.read_csv(csv_i)
# rel_energy = df['relative_energy_per_atom']

# mask = (rel_energy >= energy_limit[0]) & (rel_energy <= energy_limit[1])
# vals = rel_energy[mask].values
# idxs = df.index[mask].values

# plot_1d_scatter(vals, 
#         data_idx = idxs,
#         figsize=(10, 1),
#         xlabel=r"$E_{rel}$(meV/atom)", 
#         xlim = (0, 0.1),
#         xticks = np.linspace(0,0.11,15),
#         show_labels = True,
#         save_path=f"{dir_out}/{dir_im}/1d_graph_{file_idx}.png",
#         scale_up = 1000)
