# @title
from agox.utils.plot.colors import Colors
from agox.utils.plot import plot_atoms, plot_cell
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.constraints import FixAtoms
from typing import Optional # Import Optional
from numpy.typing import NDArray # Import NDArray

from typing import Optional, Sequence
import numpy as np
from numpy.typing import NDArray
from ase import Atoms


def plot_structure(
    atoms: Atoms,
    plane: str = 'yz+',
    plot_constraint: bool = True,
    figsize: tuple[int, int] = (10, 10),
    darken_symbols: Optional[Sequence[str]] = None,
    environment=None,
    save_path: Optional[str] = None,
    radius_factor: float = 0.8,
    repeat: int = 1,
    cell_offset: NDArray[np.floating] = np.array([0.0, 0.0, 0.0]),
    set_axis_off: bool = True,
    add_cell: bool = True,
    linewidths_cell: float = 1.5,
    linewidths_environment: float = 1.0,
    plot_show: bool = True,
    darken_factor: float = 0.4,
    n_darken_layers: int = 2,
) -> None:
    """
    Plot an ASE atomic structure with customized colors, optional confinement,
    and selective darkening of lower monolayers.

    Parameters
    ----------
    atoms
        ASE Atoms object to plot.
    plane
        Viewing plane ('xy+', 'yz+', 'xz+').
    plot_constraint
        Whether to visualize atomic constraints.
    figsize
        Size of the matplotlib figure.
    darken_symbols
        List of chemical symbols whose lowest layers should be darkened.
    environment
        Optional confinement/environment object providing a cell.
    save_path
        If given, save the figure to this path.
    radius_factor
        Scaling factor for atomic radii in the plot.
    repeat
        Unit-cell repetition factor for visualization.
    cell_offset
        Offset applied when drawing periodic images of the cell.
    set_axis_off
        Whether to hide plot axes.
    add_cell
        Whether to draw the simulation cell.
    linewidths_cell
        Line width for the main cell outline.
    linewidths_environment
        Line width for the environment/confinement cell.
    plot_show
        Whether to display the plot interactively.
    darken_factor
        Color darkening factor for selected atoms.
    n_darken_layers
        Number of layers to split into along z; the lowest is darkened.
    """

    atom_colors = Colors(atoms)

    # --- Base coloring by element -------------------------------------------
    oxygen_indices = [a.index for a in atoms if a.symbol == 'O']
    atom_colors.set_color('red', indices=oxygen_indices)
    atom_colors.lighten(indices=oxygen_indices, factor=0.2)

    magnesium_indices = [a.index for a in atoms if a.symbol == 'Mg']
    atom_colors.set_color('orange', indices=magnesium_indices)

    iron_indices = [a.index for a in atoms if a.symbol == 'Fe']
    atom_colors.set_color('green', indices=iron_indices)

    # --- Darken lowest monolayer(s) for selected symbols ----------------------
    if darken_symbols:
        # Atom indices matching any requested symbol
        target_indices = [
            a.index for a in atoms
            if a.symbol in darken_symbols
        ]

        # Corresponding z-coordinates
        z_positions = np.array(
            [atoms[i].position[2] for i in target_indices]
        )

        # Sort atoms from lowest to highest along z
        sort_order = np.argsort(z_positions)
        sorted_indices = [target_indices[i] for i in sort_order]

        # Split into layers by atom count (assumes similar monolayer sizes)
        layers = np.array_split(sorted_indices, n_darken_layers)

        # Darken atoms in the lowest layer only
        lowest_layer_indices = layers[0]
        for atom_index in lowest_layer_indices:
            atom_colors.darken(indices=atom_index, factor=darken_factor)

    # --- Plot setup ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    plot_atoms(
        ax,
        atoms,
        colors=atom_colors,
        plane=plane,
        radius_factor=radius_factor,
        plot_constraint=plot_constraint,
        patch_kwargs=dict(linewidth=1.0),
        repeat=repeat,
    )

    # --- Draw simulation cell ------------------------------------------------
    if add_cell:
        plot_cell(
            ax,
            atoms.cell,
            plane=plane,
            collection_kwargs=dict(
                linewidths=linewidths_cell,
                linestyles='--',
                dashes=(0, (5, 10)),
            ),
        )

        plot_cell(
            ax,
            atoms.cell,
            plane=plane,
            offset=cell_offset,
            collection_kwargs=dict(linewidths=0),
        )

        plot_cell(
            ax,
            atoms.cell,
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

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if plot_show:
        plt.show()

    plt.close()

# Example usage 2:

# from ase.constraints import FixAtoms
# from scripts.plot_structure import plot_structure
# import pandas as pd

# energy_limit = (0, 0.1) # in eV
# energy_limit_between = 0.0001 # greater than this
# csv_i = f"{dir_out}/data_0.csv"

# df = pd.read_csv(csv_i)
# db_path = f"{dir_path}/1_db/db_0.db"
# db = Database(filename=db_path)
# db.restore_to_memory()
# traj = db.restore_to_trajectory()

# rel_e_per_atom = df['relative_energy_per_atom']
# mask = (rel_e_per_atom >= energy_limit[0]) & (rel_e_per_atom <= energy_limit[1])
# vals = rel_e_per_atom[mask].values
# idxs = df.index[mask].values

# filtered_idxs = []

# # prev = vals[order[0]] # below any energy
# prev = 0 # below any energy
# for v, i in zip(vals, idxs):
#     if v >= prev + energy_limit_between or v == 0:
#         filtered_idxs.append(i)
#         prev = v

# if "feonmgo" in dir_path.lower():
#     substrate = ['Mg', 'O']
# elif "mgoonfe" in dir_path.lower():
#     substrate = ['Fe']
# else:
#     substrate = None

# os.makedirs(f"{dir_out}/{dir_im}/{file_idx}", exist_ok=True)

# cell_offset = np.array([1, 1, 0.0]) # Example offset
# for struct_idx in filtered_idxs:
#     struct_i = traj[struct_idx]

#     fixed_indices = [atom.index for atom in struct_i if atom.symbol in substrate]
#     struct_i.set_constraint(FixAtoms(indices=fixed_indices))
    
#     # to_keep = [atom.index for atom in struct_i if atom.symbol not in substrate]
#     # struct_i = struct_i[to_keep]
    
#     cell_offset = np.array([0.5, 0.5, 0.0])
#     for plane in ["xy+", "yz+", "xz+"]:
#         plot_structure(
#             struct_i,
#             plane=plane,
#             save_path=f"{dir_out}/{dir_im}/{file_idx}/figure_{plane}_{struct_idx}.png",
#             figsize=(5, 5),
#             repeat=5,
#             cell_offset=cell_offset,
#             plot_show=True,
#         )

# Example usage:
# from ase.build import surface, bulk
# from ase import Atoms
# from ase.io import write, read
# from ase.constraints import FixAtoms

# from agox.environments import Environment
# from agox.samplers import FixedSampler
# from agox.generators import RattleGenerator

# from scripts.build_mgo_stack import build_mgo_stack
# from scripts.build_fe_stack import build_fe_stack
# from scripts.hetero_struct_randomize import HeteroStructRandomize
# # from scripts.plot_structure import plot_structure


# # ---------------------------------------------------------------------------
# # Physical and structural parameters
# # ---------------------------------------------------------------------------
# vacuum = 10.0            # Vacuum thickness (Å)
# a_mgo = 4.212            # MgO bulk lattice constant (Å)
# a_fe = 2.870190          # Optimized Fe lattice constant (Å)
# dist_z_fe2o = 2.3        # Fe–O interface separation (Å)

# num_layers_mgo = 5       # Number of MgO layers
# num_layers_fe = 5       # Number of Fe layers
# super_cell = (10, 10, 1) # Lateral supercell size


# # ---------------------------------------------------------------------------
# # Output paths
# # ---------------------------------------------------------------------------
# path_xsf = f"{dir_out}/0_template"
# path_fig = f"{dir_out}/{dir_im}/generate_struc"

# os.makedirs(path_xsf, exist_ok=True)
# os.makedirs(path_fig, exist_ok=True)


# # ---------------------------------------------------------------------------
# # Read structure and apply constraints
# # ---------------------------------------------------------------------------
# structs = read(f"{dir_out}/{dir_xsf}/0/struct_0.xsf")

# # Fix substrate atoms (MgO)
# substrate_indices = [
#     atom.index for atom in structs
#     if atom.symbol in ['Mg', 'O']
# ]
# structs.set_constraint(FixAtoms(indices=substrate_indices))


# # ---------------------------------------------------------------------------
# # Visualization settings
# # ---------------------------------------------------------------------------
# cell_offset = np.array([1.0, 1.0, 0.0])   # Offset for periodic image drawing

# darken_symbols = ['Fe']   # Elements whose lowest layers are darkened
# n_darken_layers = 2       # Number of layers to split along z


# # ---------------------------------------------------------------------------
# # Plot structure
# # ---------------------------------------------------------------------------
# plot_structure(
#     structs,
#     plane='zy+',           # Viewing direction
#     save_path="./image_xy.png",
#     figsize=(5, 5),
#     repeat=4,
#     add_cell=True,
#     plot_show=True,
#     cell_offset=cell_offset,
#     darken_symbols=darken_symbols,
#     darken_factor=0.3,
#     n_darken_layers=n_darken_layers,
# )
