from ase.io import write
import os
import numpy as np
import pandas as pd
from agox.databases import Database
from .plot_structure import plot_structure
from .plot_1d_scatter import plot_1d_scatter
from .plot_distance_distributions import plot_distance_distributions
from .calculate_relative_energy import calculate_relative_energy

def process_database(
        db_path: str = None,
        file_idx: int = None,
        num_of_min: int = None,
        energy_range: (float,float) = None,
        dir_out: str = None,
        dir_xsf_traj: str = None,
        dir_xsf: str = None, 
        is_per_atom: bool = True,
        plot_1d_xlim: tuple = (0, 1)
    ):
    """
    Load a trajectory database, extract energies, compute relative energies,
    identify minimum energy structures, and save associated plots and files.

    Parameters
    ----------
    db_path : str
        Path to the database file.
    file_idx : int
        Index used for labeling output.
    num_of_min : int
        Maximum number of minimum energy structures to export.
    dir_out : str
        Root output directory.
    dir_xsf_traj : str
        Directory to store full trajectories in XSF and TRAJ format.
    dir_xsf : str
        Directory to store selected structures.
    is_per_atom : bool, optional
        Normalize energies per atom if True.
    plot_1d_xlim : tuple, optional
        X axis limits for 1D scatter plot.
    """

    __version__ = "0.0.2"

    # Load the database fully into memory
    database = Database(filename=db_path)
    database.restore_to_memory()
    trajs = database.restore_to_trajectory()

    print(f"Loaded {file_idx} from Database")

    # Prepare output directories
    os.makedirs(f"{dir_out}/{dir_xsf}/{file_idx}", exist_ok=True)
    path_fig = f"{dir_out}/7_fig"
    os.makedirs(path_fig, exist_ok=True)

    path_fig_atom = f"{path_fig}/0_fig_atom"
    os.makedirs(path_fig_atom, exist_ok=True)
    os.makedirs(f"{path_fig_atom}/{file_idx}", exist_ok=True)

    # Save full trajectory in two formats for convenience
    write(f"{dir_out}/{dir_xsf_traj}/traj_{file_idx}.xsf", trajs)
    write(f"{dir_out}/{dir_xsf_traj}/traj_{file_idx}.traj", trajs)

    # Collect potential energies
    all_e = [atoms_frame.get_potential_energy() for atoms_frame in trajs]
    
    # Compute relative energies
    sorted_e = np.argsort(all_e)
    all_e = [all_e[i] for i in sorted_e]
    min_e = all_e[sorted_e[0]]


    if is_per_atom:
        e_new = [(e - min_e) / len(atoms_i) for e, atoms_i in zip(all_e, trajs)]
        x_label = r"$E$ Rel. (eV/atom)"
    else:
        e_new = [e - min_e for e in all_e]
        x_label = r"$E$ Rel. (eV)"

    # Export the absolute minimum energy structure
    # struct_min = trajs[sorted_e[0]].copy()
    struct_min = trajs[sorted_e[0]].copy()
    for key in ["initial_magmoms", "magmoms"]:
        if key in struct_min.arrays:
            del struct_min.arrays[key]

    write(f"{dir_out}/{dir_xsf}/{file_idx}/struct_min_{sorted_e[0]}.xsf", struct_min)

    # Write summary information
    with open(f"{dir_out}/e_note.md", "a") as file:
        scheme = db_path.split("/")[-3]
        file.write(f"\n=== Scheme: {scheme} | Index: {file_idx} | Total Structures: {len(trajs)} ===\n")
        file.write("idx\tEtotal (eV)\tE_relative (eV)\n")
        file.write(f"{db_path}\n")

        if energy_range:
            low, high = energy_range
            list_new = [i for i, e in enumerate(e_new) if low <= e <= high]
            
        elif num_of_min:
            list_new = e_new[:num_of_min]
            list_new = [list_new[i] for i in range(len(list_new)) if list_new[i] != list_new[i - 1] - 1]

        print(f"{list_new=}")
        # Generate 1D energy scatter
        e_new_plot = [e_new[i] for i in list_new]
        plot_1d_scatter(
            e_new_plot,
            xlabel=x_label,
            xlim_low=plot_1d_xlim[0],
            xlim_high=plot_1d_xlim[1],
            xticks=np.arange(0, plot_1d_xlim[1] + 0.1, 0.25),
            save_path=f"{path_fig_atom}/axis_{file_idx}.png",
            save_data_path=f"{path_fig_atom}/data_{file_idx}.csv"
        )

        # Export each selected minimum structure with plots
        cell_offset = np.array([0.5, 0.5, 0.0])

        for j, i in enumerate(list_new):
            file.write(f"{i}\t{all_e[i]:.6f}\t{e_new[i]:.6f}\n")

            atoms_i = trajs[i].copy()
            for key in ["initial_magmoms", "magmoms"]:
                if key in atoms_i.arrays:
                    del atoms_i.arrays[key]

            write(f"{dir_out}/{dir_xsf}/{file_idx}/struct_{i}_{j}.xsf", atoms_i)

            for plane in ["xy+", "yz+", "xz+"]:
                plot_structure(
                    atoms_i,
                    plane=plane,
                    save_path=f"{path_fig_atom}/{file_idx}/figure_{plane}_{i}_{j}.png",
                    figsize=(5, 5),
                    repeat=5,
                    cell_offset=cell_offset,
                    plot_show=False,
                )
