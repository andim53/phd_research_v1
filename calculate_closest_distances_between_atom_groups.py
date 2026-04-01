from ase.geometry import get_distances
import numpy as np

def calculate_closest_distances_between_atom_groups(atoms, group1_symbols, group2_symbols, cutoff=3.0):
    """
    Calculates the closest distance for each atom in group1 to any atom in group2,
    and returns the indices of both atoms involved in the distance.

    Parameters:
        atoms (Atoms): The ASE Atoms object representing the structure.
        group1_symbols (list or str): A list of element symbols or a single symbol for atoms in the first group.
        group2_symbols (list or str): A list of element symbols or a single symbol for atoms in the second group.
        cutoff (float, optional): The maximum distance to consider for finding the closest neighbor.
                                  If None, all distances are considered.

    Returns:
        list: A list of tuples, where each tuple contains (closest_distance, atom_index_in_group1, atom_index_in_group2).
              Returns (np.inf, atom_index_in_group1, None) if no neighbor is found within the cutoff.
    """

    __version__ = "0.1"

    # Convert str to list
    if isinstance(group1_symbols, str):
        group1_symbols = [group1_symbols]
    if isinstance(group1_symbols, str):
        group2_symbols = [group2_symbols]
    
    closest_distances_with_indices = []

    group1_indices = [i for i, atom in enumerate(atoms) if atom.symbol in group1_symbols]
    group2_indices = [i for i, atom in enumerate(atoms) if atom.symbol in group2_symbols]

    # Avoid repeating pairs
    seen_pairs = set()

    for idx1 in group1_indices:
        min_dist = np.inf
        min_dist_idx2 = None

        for idx2 in group2_indices:
            if idx1 == idx2:
                continue

            # enforce unique pair ordering to avoid repetition
            pair = tuple(sorted((idx1, idx2)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            dist = atoms.get_distance(idx1, idx2, mic=True)
            if cutoff is None or dist <= cutoff:
                if dist < min_dist:
                    min_dist = dist
                    min_dist_idx2 = idx2

        # # Only keep valid results (exclude np.inf or None)
        # if np.isfinite(min_dist) and min_dist_idx2 is not None:
        
        closest_distances_with_indices.append((min_dist, idx1, min_dist_idx2))

    return closest_distances_with_indices

