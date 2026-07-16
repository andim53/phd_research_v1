import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Dict, Any

from agox.databases import Database

from .calculate_relative_energy import calculate_relative_energy
from .calculate_best_so_far import calculate_best_so_far

from matplotlib.ticker import FixedLocator

# from scripts.calculate_relative_energy import calculate_relative_energy
# from scripts.calculate_best_so_far import calculate_best_so_far

def plot_energy_progression(
    db_paths: List[Tuple[str, int]],
    labels: List[str],
    dir_out: str,
    dir_im: str,
    figsize: Tuple[float, float] = (5, 4),
    add_scatter: bool = False,
    y_label: str = r'$E_{rel}$ (eV/atom)',
    x_label: str = r'Evaluated Candidate Count ($N_i$)',
    line_styles: Sequence[str] = (':', '--', '-', '-.'),
    scatter_markers: Sequence[str] = ('o', 's', '^', 'd'),

    linewidth = 1.8,
    colors: Optional[Sequence[str]] = ["#1B70FC", "#E31A1C", "#33A02C", "#FF7F00"],
    x_lim: Tuple[Optional[float], Optional[float]] = (None, None),
    y_lim: Tuple[Optional[float], Optional[float]] = (None, None),
    
    rcParams: Dict[str, Any] = {
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.edgecolor': 'black',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.top': True,
        'ytick.right': True,
        'legend.frameon': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
    },

    xticks = [None],
    yticks = [None],
    
    index_skip: int = 2,
    scale_up: Optional[float] = None,
) -> None:
    """
    Plot best-so-far progression of relative energies from multiple databases.

    Args:
        db_paths: List of (dir_path, file_idx) tuples.
        labels: Text labels for each dataset.
        dir_out: Directory for output files.
        dir_im: Folder name for saving image.
        figsize: Figure size in inches.
        add_scatter: If True, display markers at selected indices.
        y_label: y-axis label.
        line_styles: Line styles used per dataset.
        scatter_markers: Markers used when add_scatter=True.
        rcParams: Parameters applied to matplotlib before plotting.
        x_lim: (min, max) x-axis limits.
        y_lim: (min, max) y-axis limits.
        colors: Optional list of line colors.
        index_skip: Minimum index spacing for scatter markers.
        scale_up: Optional scaling factor for y tick labels.
    """

    plt.rcParams.update(rcParams)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    for i, (dir_path, _) in enumerate(db_paths):
        # Database loading
        db_path = Path(dir_path) / "1_db" / "db_0.db"
        print(db_path)
        database = Database(filename=str(db_path))
        trajs = database.restore_to_trajectory()

        # Data Calculation
        _, _, _, rel_energies = calculate_relative_energy(trajs)
        best_energies, best_indices = calculate_best_so_far(rel_energies)

        color = colors[i % len(colors)] if colors else 'black'

        # Plot Main Line
        ax.plot(
            range(len(best_energies)),
            best_energies,
            linewidth=linewidth,
            linestyle=line_styles[i % len(line_styles)],
            color=color,
            label=labels[i],
            zorder=3
        )

        if add_scatter:
            # Filtering indices for cleaner scatter
            filt_idx = [best_indices[0]]
            for idx in best_indices[1:]:
                if idx > filt_idx[-1] + index_skip:
                    filt_idx.append(idx)
            
            ax.scatter(
                filt_idx, [best_energies[k] for k in filt_idx],
                color=color, s=15, zorder=4, edgecolor='white', linewidth=0.5
            )

    # Axis Formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if xticks[0] != None :
        plt.xticks(xticks, fontsize=12) 
        ax.set_xmargin(0)
        xticks_minor = (xticks[:-1] + xticks[1:]) / 2
        ax.xaxis.set_minor_locator(FixedLocator(xticks_minor))
    if yticks[0] != None :
        plt.yticks(yticks, fontsize=12)
        ax.set_ymargin(0)
        yticks_minor = (yticks[:-1] + yticks[1:]) / 2
        ax.yaxis.set_minor_locator(FixedLocator(yticks_minor))

    # Set Limits
    if any(v is not None for v in x_lim): ax.set_xlim(x_lim)
    if any(v is not None for v in y_lim): ax.set_ylim(y_lim)

    if scale_up:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{y * scale_up:.0f}")
        )

    # ax.legend(loc='upper right')
    ax.legend(loc='best', fontsize=10)

    # Save Logic
    output_dir = Path(dir_out) / dir_im
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    fig_path = output_dir / "relative_energy_progression.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


'''

from agox.databases import Database
from ase.io import write
import os
import numpy as np
from scripts.plot_energy_progression import plot_energy_progression

# dir_colab = "/content/drive/MyDrive/phd_research/results"
# dir_simul = f"{dir_colab}/0_simul"
# dir_out = f"{dir_colab}/0_analy"

dir_simul = "0_result"          # Directory for resulted simulations
dir_out = "0_analy"             # Directory for the analysis

dir_xsf_traj = "0_xsf_traj"     # Stores entire trajectory
dir_xsf = "1_xsf"               # Stores all structures
dir_im = '2_im'

os.makedirs(f"{dir_out}", exist_ok=True)
os.makedirs(f"{dir_out}/{dir_xsf_traj}", exist_ok=True)
os.makedirs(f"{dir_out}/{dir_xsf}", exist_ok=True)
os.makedirs(f"{dir_out}/{dir_im}",exist_ok=True)


db_paths = [
    (f"{dir_simul}/1_1ML_feOnMgo", 0),
    (f"{dir_simul}/2_1ML_mgoOnFe", 1),
]

labels = [
    "Fe on MgO",
    "MgO on Fe",
]

colors = [
    "#1B70FC",  # Blue
    "#E31A1C",  # Red
]

rcParams = {
    'font.size': 12,
    'font.family': 'sans-serif',

    'axes.linewidth': 1.5,
    'axes.edgecolor': 'black',

    'axes.spines.top': True,
    'axes.spines.right': True,

    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'xtick.major.size': 10,
    'ytick.major.size': 10,
    'xtick.minor.size': 5,      # Added minor tick size
    'ytick.minor.size': 5,      # Added minor tick size
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1.0,    # Added minor tick width
    'ytick.minor.width': 1.0,    # Added minor tick width

    'xtick.top': False,
    'ytick.right': False,         # Enabled right ticks for a "boxed" look

    'axes.grid': False,
    'legend.frameon': True,
}

xticks = np.linspace(0, 110, 7)
xticks = np.round(xticks).astype(int)

yticks = np.linspace(0, 1.3, 10)
yticks = np.round(yticks, 2)

x_lim=(0, 110)
y_lim=(0, 1.3)


plot_energy_progression(
    db_paths,
    labels,
    dir_out,
    dir_im,

    figsize=(5, 4),

    y_label=r'$E_{rel}$ (eV/atom)',
    x_label=r'Evaluated Candidate Count ($N_i$)',
    line_styles=['-'],

    rcParams=rcParams,
    colors=colors,

    linewidth=2.0,

    x_lim=x_lim,
    y_lim=y_lim,

    xticks = xticks,
    yticks = yticks,

    # index_skip=3,
    # add_scatter=True,
    # scale_up = True,
)

'''