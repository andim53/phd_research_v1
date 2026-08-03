from typing import Optional, Sequence
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from ase import Atoms
from ase.constraints import FixAtoms
from agox.utils.plot.colors import Colors
from agox.utils.plot import plot_atoms, plot_cell

def plot_structure(
    atoms: Atoms,
    plane: str = 'yz+',
    constraint_symbols: Optional[Sequence[str]] = None,
    height_darken_symbols: Optional[Sequence[str]] = None, # Kept for backward compatibility but unused
    figsize: tuple[int, int] = (10, 10),
    environment=None,
    save_path: Optional[str] = None,
    radius_factor: float = 0.8,
    repeat: int = 1,
    cell_offset: NDArray[np.floating] = np.array([0.0, 0.0, 0.0]),
    set_axis_off: bool = True,
    add_cell: bool = True,
    linewidths: float = 1.5,
    linewidths_environment: float = 1.0,
    plot_show: bool = True,
    darken_factor: float = 0.4,
    n_darken_layers: int = 2,
    show_colorbar: bool = False,
    cbar_label: str = r'$\Delta z$ ($\AA$)',
    max_cbar: float = 5.0,
    num_ticks: int = 3,
    fontsize: int = 10
) -> None:
    """
    Plot an ASE atomic structure where unconstrained cluster atoms get 
    progressively darker with height. If multiple unconstrained elements exist,
    multiple colorbars are rendered.
    """
    # Make a copy to avoid mutating the original object
    atoms_copy = atoms.copy()
    
    # Apply dynamic element-based constraints if specified
    if constraint_symbols:
        fixed_indices = [a.index for a in atoms_copy if a.symbol in constraint_symbols]
        atoms_copy.set_constraint(FixAtoms(indices=fixed_indices))

    # --- Base coloring framework -------------------------------------------
    atom_colors = Colors(atoms_copy)

    oxygen_indices = [a.index for a in atoms_copy if a.symbol == 'O']
    if oxygen_indices:
        atom_colors.set_color('red', indices=oxygen_indices)
        atom_colors.lighten(indices=oxygen_indices, factor=0.2)

    magnesium_indices = [a.index for a in atoms_copy if a.symbol == 'Mg']
    if magnesium_indices:
        atom_colors.set_color('orange', indices=magnesium_indices)

    iron_indices = [a.index for a in atoms_copy if a.symbol == 'Fe']
    if iron_indices:
        atom_colors.set_color('green', indices=iron_indices)

    # --- Identify Unconstrained Atoms and Elements -------------------------
    if constraint_symbols:
        free_indices = [a.index for a in atoms_copy if a.symbol not in constraint_symbols]
    else:
        free_indices = [a.index for a in atoms_copy]

    # Map standard element strings to colors
    element_color_map = {
        'Fe': 'green',
        'Mg': 'orange',
        'O': 'red'
    }

    unique_free_elements = []
    if free_indices:
        free_z = np.array([atoms_copy[idx].position[2] for idx in free_indices])
        z_min = np.min(free_z)
        delta_z = np.max(free_z) - z_min
        print(f"Calculated unconstrained cluster Delta Z: {delta_z:.4f} Å")
        
        # Get unique elements ordered by discovery in the unconstrained indices
        for idx in free_indices:
            sym = atoms_copy[idx].symbol
            if sym not in unique_free_elements:
                unique_free_elements.append(sym)
    else:
        delta_z = 0.0
        z_min = 0.0
        print("Warning: No unconstrained atoms found to evaluate delta z.")

    # --- Progressive Continuous Darkening Logic ----------------------------
    if free_indices:
        for idx in free_indices:
            sym = atoms_copy[idx].symbol
            base_cluster_color = element_color_map.get(sym, 'green')
            base_rgba = np.array(mcolors.to_rgba(base_cluster_color))
            
            # Distance from cluster base
            atom_height = atoms_copy[idx].position[2] - z_min 
            norm_height = np.clip(atom_height / max_cbar, 0.0, 1.0)
            
            # Calculate darkening multiplier
            shade_factor = 1.0 - (norm_height * darken_factor)
            scaled_rgb = base_rgba[:3] * shade_factor
            hex_color = mcolors.to_hex(scaled_rgb)
            
            atom_colors.set_color(hex_color, indices=[idx])

    # --- Plot setup ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    has_constraints = len(atoms_copy.constraints) > 0

    plot_atoms(
        ax,
        atoms_copy,
        colors=atom_colors,
        plane=plane,
        radius_factor=radius_factor,
        plot_constraint=has_constraints,
        patch_kwargs=dict(linewidth=1.0),
        repeat=repeat,
    )

    # --- Draw simulation cell ------------------------------------------------
    if add_cell:
        plot_cell(
            ax,
            atoms_copy.cell,
            plane=plane,
            collection_kwargs=dict(
                linewidths=linewidths,
                linestyles='--',
                dashes=(0, (5, 10)),
            ),
        )

        plot_cell(
            ax,
            atoms_copy.cell,
            plane=plane,
            offset=cell_offset,
            collection_kwargs=dict(linewidths=0),
        )

        plot_cell(
            ax,
            atoms_copy.cell,
            plane=plane,
            offset=-cell_offset,
            collection_kwargs=dict(linewidths=0),
        )

    # --- Optional confinement/environment cell ------------------------------
    if environment:
        plot_cell(
            ax,
            environment.get_confinement_cell(),
            plane=plane,
            offset=environment.get_confinement_corner(),
            collection_kwargs=dict(
                linewidths=linewidths_environment,
                edgecolors='red',
                linestyles='dashed',
            ),
        )

    if set_axis_off:
        ax.set_axis_off()

    # --- Generation of Multi-Element Colorbars ------------------------------
    if show_colorbar and unique_free_elements:
        num_cbars = len(unique_free_elements)
        
        # Sequentially place each colorbar to avoid overlap
        for i, sym in enumerate(unique_free_elements):
            base_cluster_color = element_color_map.get(sym, 'green')
            base_rgba = np.array(mcolors.to_rgba(base_cluster_color))
            
            darkened_rgba = base_rgba.copy()
            darkened_rgba[:3] *= (1.0 - darken_factor)
            
            # Custom linear map matching the element's darkening profile
            atom_shaded_cmap = mcolors.LinearSegmentedColormap.from_list(
                f'progressive_{sym}_darken', 
                [base_rgba, darkened_rgba], 
                N=256
            )
            
            norm = mcolors.Normalize(vmin=0.0, vmax=max_cbar)
            sm = plt.cm.ScalarMappable(cmap=atom_shaded_cmap, norm=norm)
            sm.set_array([])
            
            # Calculate shifting padding locations if multiple bars are needed
            # First bar gets standard pad, subsequent bars are pushed right
            pad_shift = 0.05 + (i * 0.15) 
            
            # Create the colorbar axis anchor
            cbar = fig.colorbar(sm, ax=ax, pad=pad_shift, shrink=0.55, aspect=20)
            
            # Label includes the element symbol to easily identify the colorbar
            label_text = f"{sym} {cbar_label}" if num_cbars > 1 else cbar_label
            cbar.set_label(label_text, fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize - 2)
            
            ticks = np.linspace(0.0, max_cbar, num_ticks)
            cbar.set_ticks(ticks)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if plot_show:
        plt.show()

    plt.close()