import numpy as np
from ase import Atoms

def replace_slab_position(slab: Atoms, original_pos, new_pos, tol=1e-3):
    """
    Moves an atom from original_pos to new_pos in the given ASE Atoms object.

    Parameters:
        slab (ase.Atoms): The atomic structure to modify.
        original_pos (array-like): The original position of the atom to move.
        new_pos (array-like): The new position to move the atom to.
        tol (float): Tolerance for position matching (default: 1e-3).

    Returns:
        ase.Atoms: Modified Atoms object with the atom moved.
    """
    original_pos = np.array(original_pos)
    new_pos = np.array(new_pos)

    for i, atom in enumerate(slab):
        if np.allclose(atom.position, original_pos, atol=tol):
            slab.positions[i] = new_pos
            return slab

    raise ValueError("No atom found at the specified original position.")
