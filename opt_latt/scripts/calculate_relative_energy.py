from typing import Iterable, List, Tuple
from ase.atoms import Atoms

def calculate_relative_energy(traj: Iterable[Atoms]) -> Tuple[List[float], float, List[float], List[float]]:
    """
    Compute total energies, minimum energy, relative energies, and
    relative energies per atom for a trajectory.

    Parameters:
        traj: Iterable of Atoms objects.

    Returns:
        energies: Total energies for each frame.
        min_energy: Minimum total energy.
        rel_energy: Energy difference to the minimum for each frame.
        rel_energy_per_atom: Relative energy normalized per atom.
    """
    __version__ = 0.2
    energies = [frame.get_potential_energy() for frame in traj]
    min_energy = min(energies)
    rel_energy = [e - min_energy for e in energies]
    rel_energy_per_atom = [e / len(frame) for e, frame in zip(rel_energy, traj)]

    return energies, min_energy, rel_energy, rel_energy_per_atom


# import numpy as np

# def calculate_relative_energy(all_e, is_percentage: bool = False, is_per_atom: bool = True, atoms=None, unit: str = "eV/atom"):
#     """
#     Calculates normalized relative energies from a list of total energies.

#     Parameters:
#         all_e (list or ndarray): List or array of total energy values.

#     Returns:
#         list: List of normalized relative energies.
#     """

#     if is_percentage:
#         min_energy = np.nanmin(all_e)
#         # Use a small epsilon to avoid division by zero if min_energy is exactly 0
#         normalization_factor = abs(min_energy) if abs(min_energy) > 1e-9 else 1.0
#         rel_energies = [((e - min_energy) / normalization_factor) * 100 if not np.isnan(e) else np.nan for e in all_e]
#     else:
#         sorted_e = np.argsort(all_e)
#         min_e = all_e[sorted_e[0]]

#         if is_per_atom: 
#             atom_num = len(atoms[0])
#             print(f"{atom_num=}")
#             rel_energies = [(e - min_e)/len(atom_i) for e, atom_i in zip(all_e, atoms)]
            
#         else:
#             rel_energies = [e - min_e for e in all_e]
    

#             # unit conversion
#         if unit == "meV/atom" and is_per_atom and not is_percentage:
#             # convert eV/atom → meV
#             rel_energies = [(v * 1000) for v in rel_energies]
#         return rel_energies
