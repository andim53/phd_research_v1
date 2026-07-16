import random
import numpy as np
from ase import Atoms


def generate_interstitial_alloy(
    host_unit: Atoms,
    interstitial_element: str = "B",
    supercell_dim: tuple = (2, 2, 2),
    num_to_remove: int = 0,
    lattice_type: str = "fcc",
    seed: int = None,
) -> Atoms:
    """
    Generates an interstitial alloy by placing guest atoms into the octahedral
    sites of a conventional FCC or BCC metal host supercell, with random defect pruning.

    Parameters:
    -----------
    host_unit : ase.Atoms
        Conventional unit cell of the host metal.
    interstitial_element : str, default 'B'
        Symbol of the interstitial species to insert (e.g., 'B' or 'P').
    supercell_dim : tuple of ints, default (2, 2, 2)
        Supercell expansion dimensions (nx, ny, nz).
    num_to_remove : int, default 0
        Exact number of interstitial atoms to randomly prune from the system.
    lattice_type : str, default 'fcc'
        Lattice symmetry of the host unit cell ('fcc' or 'bcc').
    seed : int, optional
        Seed for the random selection generator to ensure reproducibility.

    Returns:
    --------
    ase.Atoms
        The combined host and pruned interstitial alloy structure.
    """
    if seed is not None:
        random.seed(seed)

    # --- 1. Expand Host Unit Cell to Supercell ---
    supercell = host_unit * supercell_dim
    cell_matrix = supercell.get_cell()

    # --- 2. Map Octahedral Interstitial Sites ---
    lattice_type_lower = lattice_type.lower()
    if lattice_type_lower == "fcc":
        # Fractional basis coordinates for octahedral sites in a conventional FCC cell
        octa_fractional_basis = [
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ]
    elif lattice_type_lower == "bcc":
        # Fractional basis coordinates for octahedral sites in a conventional BCC cell
        # Consists of 3 edge midpoints and 3 face centers
        octa_fractional_basis = [
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    else:
        raise ValueError("lattice_type must be either 'fcc' or 'bcc'")

    interstitial_frac_coords = []
    nx, ny, nz = supercell_dim

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for base in octa_fractional_basis:
                    frac = np.array(
                        [
                            (base[0] + i) / nx,
                            (base[1] + j) / ny,
                            (base[2] + k) / nz,
                        ]
                    )
                    interstitial_frac_coords.append(frac)

    # Map fractional positions into absolute Cartesian coordinates
    interstitial_cart_coords = np.dot(interstitial_frac_coords, cell_matrix)

    # --- 3. Apply Random Interstitial Pruning ---
    total_interstitials = len(interstitial_cart_coords)
    if num_to_remove > total_interstitials:
        raise ValueError(
            f"Requested to remove {num_to_remove} atoms, but only {total_interstitials} exist."
        )

    keep_count = total_interstitials - num_to_remove
    keep_indices = random.sample(range(total_interstitials), keep_count)
    final_interstitial_coords = interstitial_cart_coords[keep_indices]

    # --- 4. Assemble Alloy and Print Summary ---
    num_interstitials_left = len(final_interstitial_coords)
    interstitial_atoms = Atoms(
        symbols=interstitial_element * num_interstitials_left,
        positions=final_interstitial_coords,
    )

    final_alloy = supercell + interstitial_atoms

    print(f"--- System Generation Summary ({lattice_type.upper()}) ---")
    print(f"Host Formula (Supercell): {supercell.get_chemical_formula()}")
    print(f"Supercell Dimensions: {nx}x{ny}x{nz}")
    print(f"Total Host Atoms: {len(supercell)}")
    print(f"Max Interstitial Sites: {total_interstitials}")
    print(f"Removed Interstitials: {num_to_remove} (Randomly)")
    print(f"Final Chemical Formula: {final_alloy.get_chemical_formula()}")
    print(f"---------------------------------")

    return final_alloy