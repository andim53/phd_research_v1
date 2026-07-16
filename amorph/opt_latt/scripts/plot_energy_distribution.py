import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
from scipy.stats import gaussian_kde
from matplotlib.ticker import FixedLocator

# External AGOX dependencies
from agox.databases import Database
from .calculate_relative_energy import calculate_relative_energy


def plot_energy_distribution(
        db_paths: List[Tuple[str, int]],  
        labels: List[str],               
        dir_out: str,                    
        dir_im: str,                     
        figsize: Tuple[float, float] = (4, 6),
        y_label: str = r'$E_{rel}$ (eV/atom)',
        x_label: str = 'State Density',
        colors: Optional[List[str]] = None,
        plot_single: bool = False,
        single_index: int = 0,
        linewidth: float = 2.2,
        tick_length: float = 4.0,
        font_size: int = 10,
        font_family: str = 'serif',
        x_lim: Tuple[Optional[float], Optional[float]] = (None, None),
        y_lim: Tuple[Optional[float], Optional[float]] = (None, None),
        plot_type: str = "box",
        rcParams: Optional[Dict[str, Any]] = None,
        show_legend: bool = False,
        xticks: List[float] = [None],
        yticks: List[float] = [None],
        widths: float = 0.6
    ) -> None:   
    """
    Plots energy distribution as either a boxplot or a KDE density curve.
    """

    # Configure global plot aesthetics
    if rcParams is None:
        rcParams = {
            'font.size': font_size,
            'font.family': font_family,
            'mathtext.fontset': 'cm',
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'axes.linewidth': linewidth,   
            'xtick.major.width': linewidth,  
            'ytick.major.width': linewidth,  
            'xtick.major.size': tick_length,      
            'ytick.major.size': tick_length,      
        }

    plt.rcParams.update(rcParams)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Determine subset of data to plot
    indices = [single_index] if plot_single else range(len(db_paths))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    for idx in indices:
        dir_path: str = db_paths[idx][0]
        db_file: str = os.path.join(dir_path, "1_db", "db_0.db")

        # Load AGOX database and trajectories
        database = Database(filename=db_file)
        trajs = database.restore_to_trajectory()
        _, _, _, rel_energies = calculate_relative_energy(trajs)

        database = Database(filename=db_file)
        trajs = database.restore_to_trajectory()
        _, _, _, rel_energies = calculate_relative_energy(trajs)

        rel_energies = np.array(rel_energies)
        c = colors[idx % len(colors)] if colors else "lightgray"

        if plot_type == "box":
            ax.boxplot(
                rel_energies,
                positions=[idx + 1],
                widths=widths,
                patch_artist=True,
                boxprops=dict(color='black', facecolor=c, linewidth=linewidth),
                whiskerprops=dict(color='black', linewidth=linewidth),
                capprops=dict(color='black', linewidth=linewidth),
                medianprops=dict(color='black', linewidth=linewidth),
            )
        else:
            # KDE Calculation
            ys: np.ndarray = np.sort(rel_energies)
            kde: gaussian_kde = gaussian_kde(ys)
            density_y: np.ndarray = np.linspace(ys.min(), ys.max(), 300)
            density_x: np.ndarray = kde(density_y)

            ax.plot(density_x, density_y, color=c, linewidth=linewidth)
            ax.fill_betweenx(density_y, 0, density_x, color=c, alpha=0.1)

    # Label X-axis for boxplot categories
    if plot_type == "box":
        ax.set_xticks([i + 1 for i in indices])
        ax.set_xticklabels([labels[i] for i in indices], rotation=45)
    
    # Set axis limits
    if any(v is not None for v in x_lim): 
        ax.set_xlim(x_lim)
    if any(v is not None for v in y_lim): 
        ax.set_ylim(y_lim)

    # Configure X-axis ticks and minor locators
    if xticks[0] is not None:
        xticks_array: np.ndarray = np.array(xticks)
        ax.set_xticks(xticks_array)
        ax.tick_params(axis='x', labelsize=12)
        
        # Calculate and set minor ticks halfway between major ticks
        xticks_minor: np.ndarray = (xticks_array[:-1] + xticks_array[1:]) / 2
        ax.xaxis.set_minor_locator(FixedLocator(xticks_minor))
        ax.set_xmargin(0)

    # Configure Y-axis ticks and minor locators
    if yticks[0] is not None:
        yticks_array: np.ndarray = np.array(yticks)
        ax.set_yticks(yticks_array)
        ax.tick_params(axis='y', labelsize=12)
        
        # Calculate and set minor ticks halfway between major ticks
        yticks_minor: np.ndarray = (yticks_array[:-1] + yticks_array[1:]) / 2
        ax.yaxis.set_minor_locator(FixedLocator(yticks_minor))

    if show_legend:
        ax.legend(labels=labels, frameon=True)

    # Ensure plot elements align tightly to the figure edges
    ax.set_ymargin(0)
    fig.tight_layout()

    # Determine filename based on plot style
    fname = "energy_box.png" if plot_type == "box" else "energy_density.png"
    if plot_single:
        fname = f"single_{labels[single_index]}_{fname}"

    # Save output
    out_path = os.path.join(dir_out, dir_im, fname)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()