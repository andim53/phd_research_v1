from ase.io import read, write
import numpy as np
from ase import Atoms


def build_mgo_stack(slab_fe, num_layers, dist_fe2o=2.3, dist_mgo=2.106, output_path=None, vacuum=10, rotate_system=False):
    """
    Build Fe/MgO layered structure with alternating O and Mg atoms on top of an Fe slab.

    Parameters:
        slab_fe (Atoms): ASE Atoms object of the Fe slab.
        num_layers (int): Number of MgO layers to add (each layer includes 1 O and 1 Mg).
        dist_fe2o (float): Distance from Fe top surface to first O layer (in Å).
        dist_mgo (float): Distance between alternating Mg/O layers (in Å).
        output_path (str): If given, write the final structure to this path.
        vacuum (float): Vacuum gap size in Å.
        rotate_system (bool): If True, rotates the entire final system 180 degrees 
                              so the top layers become the bottom layers.

    Returns:
        Atoms: Final structure with Fe and added MgO layers.
    """

    __version__ = "0.0.2"

    # Sort Fe atoms by z-position
    z_sorted = np.sort(np.unique(slab_fe.positions[:, 2]))
    z_top = z_sorted[-1]
    z_second = z_sorted[-2]

    # Find atoms at top and second-top layers
    fe_top_atom = next(atom for atom in slab_fe if np.isclose(atom.position[2], z_top))
    fe_second_atom = next(atom for atom in slab_fe if np.isclose(atom.position[2], z_second))

    # === First Layer ===
    pos_o = fe_top_atom.position.copy()
    pos_o[2] += dist_fe2o
    atom_o = Atoms('O', positions=[pos_o])

    pos_mg = fe_second_atom.position.copy()
    pos_mg[2] = pos_o[2]
    atom_mg = Atoms('Mg', positions=[pos_mg])

    slab = slab_fe + atom_o + atom_mg

    # Store references for stacking
    prev_o = pos_o
    prev_mg = pos_mg

    # === Add Additional Layers ===
    for i in range(1, num_layers):
        # O above previous Mg
        pos_o_new = prev_mg.copy()
        pos_o_new[2] += dist_mgo
        atom_o_new = Atoms('O', positions=[pos_o_new])

        # Mg above previous O
        pos_mg_new = prev_o.copy()
        pos_mg_new[2] += dist_mgo
        atom_mg_new = Atoms('Mg', positions=[pos_mg_new])

        slab += atom_o_new + atom_mg_new

        # Update for next layer
        prev_o = pos_o_new
        prev_mg = pos_mg_new

    # === Handle System Rotation ===
    if rotate_system:
        # Rotate 180 degrees around the X-axis ('x') to flip top and bottom
        # center='ONLY_O' ensures it flips around the geometric center of the atoms
        slab.rotate(180, 'x', center='COM')

    # Re-center the slab along the z-axis with vacuum (handles cell re-alignment after rotation)
    slab.center(vacuum=vacuum, axis=2)

    # Write output if requested
    if output_path:
        write(output_path, slab)

    return slab