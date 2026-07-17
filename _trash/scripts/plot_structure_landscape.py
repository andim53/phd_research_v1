from typing import List, Optional
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


def plot_structure_landscape(
    # Core Data Inputs
    X_eigen,
    energies,  # Array, list of arrays, or dict of arrays
    z_data=None,

    # File and Save Settings
    save_path='./',
    animate_scatter=False,
    animation_fps=20,
    gif_name='conf_space_ani.gif',

    # Plot Layout and Sizing
    figsize=(6, 3),
    wspace=0.05,
    fontsize=10,
    density_left=True,
    plot_density_only=False,

    # Energy and Density View Constraints
    e_limit=(0.0, 0.8, 5),
    fill_density=False,
    density_alpha=0.12,
    dens_line_weight=1.5,
    show_limits=True,
    custom_peak_labels: Optional[List[str]] = None,
    custom_legends=None,

    # Colorbar and Colormap Styling
    cmap='viridis',
    density_cmap='Blues',
    show_colorbar=True,
    cbar_pad=0.02,
    z_limit=(None, None, 5),
    black_seed_zero=False,
    seed_zero_top_zorder=False,

    # Evaluation Flags
    plot_z_vs_e=False,
):
    """
    Plots a multi-panel atomic structural conformation landscape, mapping 
    dimensionality-reduced structural features against relative energy profiles.

    Parameters:
    -----------
    X_eigen : array-like
        The principal/eigenvector structural components (e.g., PCA of descriptors).
    energies : array-like, list of arrays, or dict
        Potential energy values (normalized per atom). Can be parsed as multi-dataset tracks.
    z_data : array-like, optional
        Physical geometric parameter data (e.g., delta z spatial layer height variations).
    save_path : str, default='./'
        Directory path to export generated image and GIF files.
    animate_scatter : bool, defaultFalse
        Enables structural configuration accumulation sequence rendering exported as a GIF.
    animation_fps : int, default20
        Frames per second constraints mapping animation speed rates.
    gif_name : str, default'conf_space_ani.gif'
        Output filename designated for saved structural state sequence animations.
    figsize : tuple, default(6, 3)
        Dimensions controlling aspect ratio structures of the main viewport.
    wspace : float, default0.05
        Visual padding between adjacent subplots panel spaces.
    fontsize : int, default10
        Base scaling parameter driving structural label sizing variables.
    density_left : bool, defaultTrue
        Toggles position layout mapping the State Density panel to the left or right axis.
    plot_density_only : bool, defaultFalse
        Bypasses scatter generation properties entirely to output clear structural density tracks.
    e_limit : tuple, default(0.0, 0.8, 5)
        Bounds and interval subdivision counts assigned to energy landscape domains.
    fill_density : bool, defaultFalse
        Fills the region beneath the state density distribution curve paths.
    density_alpha : float, default0.12
        Opacity parameters balancing density filled track visual intensities.
    dens_line_weight : float, default1.5
        Linewidth profile assigned to the Gaussian KDE curve paths.
    show_limits : bool, defaultTrue
        Triggers peak detection loops scanning global distributions to display horizontal milestones.
    custom_peak_labels : list of str, optional
        String annotations overlaying structural properties near recognized peaks (e.g., ["Island", "Flat"]).
    custom_legends : list of str, optional
        Labels used to identify specific energy components when list inputs are parsed.
    cmap : str, default'viridis'
        Colormap profile applied over individual scatter coordinates.
    density_cmap : str, default'Blues'
        Color gradient mapping sequence layers inside multi-component density plots.
    show_colorbar : bool, defaultTrue
        Displays standard visual color bars indicating physical geometric property shifts.
    cbar_pad : float, default0.02
        Visual distance spacing out figures from tracking color bars.
    z_limit : tuple, default(None, None, 5)
        Range parameters mapping custom geometric property visibility profiles.
    black_seed_zero : bool, defaultFalse
        Forces tracking index 0 dataset profile paths to register as bold black.
    seed_zero_top_zorder : bool, defaultFalse
        Forces the initial index tracking elements to draw cleanly over overlapping layout layers.
    plot_z_vs_e : bool, defaultFalse
        Generates an extra individual scatter metric parsing correlations directly across energies and metrics.

    Returns:
    --------
    matplotlib.figure.Figure
        The configured figure element containing the finalized structural landscape layout.
    """
    min_e, max_e = e_limit[0], e_limit[1]
    eticks = np.round(np.linspace(min_e, max_e, e_limit[2]), 1)

    # --- Structure & Normalize Input Energies ---
    energy_datasets = {}
    if isinstance(energies, dict):
        energy_datasets = energies
    elif isinstance(energies, (list, tuple)) and len(energies) > 0 and isinstance(energies[0], (list, np.ndarray)):
        for i, dataset in enumerate(energies):
            label = custom_legends[i] if (custom_legends and i < len(custom_legends)) else f"Dataset {i+1}"
            energy_datasets[label] = np.asarray(dataset)
    else:
        label = custom_legends[0] if custom_legends else "State Density"
        energy_datasets[label] = np.asarray(energies)

    # --- Setup Gradient Color Palette ---
    num_curves = len(energy_datasets)
    try:
        color_gradient = plt.colormaps.get_cmap(density_cmap)
    except AttributeError:
        color_gradient = plt.cm.get_cmap(density_cmap)

    sampled_colors = []
    if num_curves > 1:
        if black_seed_zero:
            sampled_colors.append('black')
            remaining_count = num_curves - 1
            if remaining_count > 1:
                color_indices = np.linspace(0.85, 0.35, remaining_count)
                for x in color_indices:
                    sampled_colors.append(color_gradient(x))
            else:
                sampled_colors.append(color_gradient(0.60))
        else:
            color_indices = np.linspace(0.95, 0.35, num_curves)
            sampled_colors = [color_gradient(x) for x in color_indices]
    else:
        sampled_colors = ['black' if black_seed_zero else color_gradient(0.85)]

    # --- Setup Subplots Axis Framework ---
    if plot_density_only:
        width_mod = 0.55 if num_curves > 1 else 0.4
        fig, ax_dens = plt.subplots(figsize=(figsize[0] * width_mod, figsize[1]))
        ax_scat = None
        ax_l, ax_r = ax_dens, None
    else:
        ratios = [1, 2.5] if density_left else [2.5, 1]
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={'width_ratios': ratios})
        fig.subplots_adjust(wspace=wspace)
        ax_scat = ax_r if density_left else ax_l
        ax_dens = ax_l if density_left else ax_r

    # --- Scatter Plot Representation ---
    scat_anim = None
    if not plot_density_only and ax_scat is not None:
        if z_data is not None:
            vmin = z_limit[0] if z_limit[0] is not None else np.min(z_data)
            vmax = z_limit[1] if z_limit[1] is not None else np.max(z_data)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            color_src = np.asarray(z_data)
        else:
            norm = None
            color_src = 'white'

        scat_energies = next(iter(energy_datasets.values())) if num_curves > 1 else list(energy_datasets.values())[0]

        if len(X_eigen) == len(scat_energies):
            if animate_scatter:
                initial_idx = 1
                initial_colors = color_src[:initial_idx] if z_data is not None else 'white'
                sc = ax_scat.scatter(
                    X_eigen[:initial_idx], scat_energies[:initial_idx], c=initial_colors, s=25,
                    cmap=cmap if z_data is not None else None, norm=norm,
                    edgecolors='black', linewidth=0.5, alpha=0.8, zorder=2
                )
            else:
                sc = ax_scat.scatter(
                    X_eigen, scat_energies, c=color_src, s=25,
                    cmap=cmap if z_data is not None else None, norm=norm,
                    edgecolors='black', linewidth=0.5, alpha=0.8, zorder=2
                )
            
            if show_colorbar and z_data is not None:
                cbar = plt.colorbar(sc, ax=ax_scat, pad=cbar_pad)
                cbar.set_label(r'$\Delta z$ (Å)', fontsize=fontsize)
                tick_locs = np.linspace(vmin, vmax, z_limit[2])
                cbar.set_ticks(tick_locs)
                cbar.set_ticklabels([f"{t:.2f}" for t in tick_locs])
        else:
            print("Warning: X_eigen layout dims don't match base layout. Skipping scatter population.")
            animate_scatter = False

    # --- Evaluate & Generate Multiple Gaussian KDE Curves ---
    energy_grid = np.linspace(min_e, max_e, 200)
    total_density = np.zeros_like(energy_grid)

    for idx, (name, data_array) in enumerate(energy_datasets.items()):
        if len(data_array) > 1:
            kde = gaussian_kde(data_array)
            density = kde.evaluate(energy_grid)
            total_density += density
            curve_color = sampled_colors[idx]

            if idx == 0 and seed_zero_top_zorder:
                current_line_zorder = 10
                current_fill_zorder = 9
            else:
                current_line_zorder = 4
                current_fill_zorder = 3

            ax_dens.plot(density, energy_grid, color=curve_color, lw=dens_line_weight, zorder=current_line_zorder, label=name)

            if fill_density:
                ax_dens.fill_betweenx(energy_grid, 0, density, color=curve_color, alpha=density_alpha, zorder=current_fill_zorder)

    ax_dens.set_xlabel('State Density (config./eV)', fontsize=fontsize)

    if num_curves > 1:
        fig.legend(frameon=False, fontsize=fontsize-2, loc='center left', bbox_to_anchor=(1.02, 0.5))

    if not plot_density_only and ax_scat is not None:
        ax_scat.set_xlabel(r'$\psi_{1d}(a.u.)$', fontsize=fontsize)
        ax_scat.xaxis.set_minor_locator(AutoMinorLocator())
        ax_scat.set_xlim(np.min(X_eigen) - 0.1, np.max(X_eigen) + 0.1)

    if plot_density_only or density_left:
        ax_dens.set_ylabel(r'$E_{i}-E_{glob}$ (eV/atom)', fontsize=fontsize)

    # --- Automatic Peak Detection and Visualization Logic ---
    if show_limits and len(energy_datasets) > 0:
        peaks, _ = find_peaks(total_density, prominence=np.max(total_density) * 0.05)
        peaks = sorted(peaks, key=lambda idx: energy_grid[idx])
        axes_to_mark = [ax_dens] if plot_density_only else [ax_l, ax_r]

        for i, peak_idx in enumerate(peaks):
            peak_energy = energy_grid[peak_idx]

            for ax in axes_to_mark:
                if ax is not None:
                    ax.axhline(y=peak_energy, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

            if custom_peak_labels and i < len(custom_peak_labels):
                display_text = f"{custom_peak_labels[i]}: {peak_energy:.3f} eV"
            else:
                display_text = f"Peak {i+1}: {peak_energy:.3f} eV"

            t = ax_dens.text(0.05, peak_energy + 0.005, display_text, fontsize=8, color='black', verticalalignment='bottom', zorder=11)
            t.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])

    # --- Optional Correlation Plot Implementation ---
    if plot_z_vs_e and z_data is not None:
        fig_corr, ax_corr = plt.subplots(figsize=(4, 3))
        scat_energies = next(iter(energy_datasets.values()))
        
        ax_corr.scatter(z_data, scat_energies, s=20, alpha=0.7, c='white', edgecolors='black', linewidth=0.5)
        ax_corr.set_xlabel(r'$\Delta z$ (Å)', fontsize=fontsize)
        ax_corr.set_ylabel(r'$E_{i}-E_{glob}$ (eV/atom)', fontsize=fontsize)
        ax_corr.set_ylim(e_limit[0], e_limit[1])
        plt.tight_layout()
        plt.savefig(f'{save_path}/z_vs_energy_correlation.png', dpi=300)
    
    # Boundary Fixes
    ax_dens.set_ylim(min_e, max_e)
    ax_dens.set_yticks(eticks)
    ax_dens.yaxis.set_minor_locator(AutoMinorLocator())
    plt.tight_layout()

    # --- Handle Animation Generation ---
    if not plot_density_only and animate_scatter:
        total_points = len(X_eigen)
        
        def update_frame(frame):
            current_count = int((frame + 1) * (total_points / 100))
            current_count = min(max(1, current_count), total_points)
            
            offsets = np.vstack((X_eigen[:current_count], scat_energies[:current_count])).T
            sc.set_offsets(offsets)
            
            if z_data is not None:
                sc.set_array(color_src[:current_count])
            return sc,

        scat_anim = FuncAnimation(fig, update_frame, frames=100, interval=1000 // animation_fps, blit=True)
        scat_anim.save(f'{save_path}/{gif_name}', writer='pillow', fps=animation_fps)
        print(f"Animation saved successfully to: {save_path}/{gif_name}")
    else:
        plt.savefig(f'{save_path}/conf_space.png', dpi=300, bbox_inches='tight')
        
    return fig