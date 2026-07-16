from ase.io import write
import numpy as np
from .plot_structure import plot_structure


def write_best_so_far_xsf(trajs, energies, output_dir):
    """
    Writes the best-so-far structures from a list of ASE trajectories to XSF files.

    Parameters:
        trajs (list): List of ASE Atoms objects.
        energies (list): List of potential energies corresponding to trajs.
        output_dir (str): Directory to save XSF files.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    best_so_far = []
    best_idx = None
    current_best = None

    for i, e in enumerate(energies):
        if current_best is None or e < current_best:
            current_best = e
            best_idx = i
        best_so_far.append(best_idx)

    unique_best_so_far = sorted(set(best_so_far), key=best_so_far.index)

    for idx in unique_best_so_far:
        atoms = trajs[idx].copy()

        if 'initial_magmoms' in atoms.arrays:
            del atoms.arrays['initial_magmoms']
        if 'magmoms' in atoms.arrays:
            del atoms.arrays['magmoms']

        fname = os.path.join(output_dir, f"{idx}.xsf")
        write(fname, atoms)

        dir_im = f"{output_dir}/dir_im"
        os.makedirs(dir_im, exist_ok=True)

        cell_offset = np.array([1, 1, 0.0])
        plot_structure(
                    atoms,
                    plane='xy+',
                    save_path=f"{dir_im}/figure_xy_{idx}.png",
                    figsize = (5,5),
                    repeat=5,
                    cell_offset = cell_offset,
                    plot_show = False,
            )
        
        plot_structure(
                    atoms,
                    plane='yz+',
                    save_path=f"{dir_im}/figure_yz_{idx}.png",
                    figsize = (5,5),
                    repeat=5,
                    cell_offset = cell_offset,
                    plot_show = False,
            )

        plot_structure(
                    atoms,
                    plane='xz+',
                    save_path=f"{dir_im}/figure_xz_{idx}.png",
                    figsize = (5,5),
                    repeat=5,
                    cell_offset = cell_offset,
                    plot_show = False,
            )

"""
# Example
from scripts.write_best_so_far_xsf import write_best_so_far_xsf

from agox.models.descriptors.fingerprint import Fingerprint
from ase.io import read
from sklearn.decomposition import PCA

for dir_path, file_idx in db_paths:

    db_path = f"{dir_path}/1_db/db_0.db"
    database = Database(filename=db_path)
    database.restore_to_memory()
    trajs = database.restore_to_trajectory()

    all_e = []
    for atoms_frame in trajs:
        e = atoms_frame.get_potential_energy()
        all_e.append(e)

    write_best_so_far_xsf(trajs=trajs,energies=all_e, output_dir=f"{dir_out}/{dir_xsf_min_so_far}/{file_idx}")
"""