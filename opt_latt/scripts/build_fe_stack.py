from ase import Atoms
from ase.io import write
import numpy as np

def build_fe_stack(slab_fe, num_layers, vacuum=10, output_path=None):
    """
    Build a stack of Fe slabs in a zig-zag pattern (top, bottom, top-bottom, etc.).

    Parameters:
        slab_fe (Atoms): ASE Atoms object of the single Fe slab layer (should contain at least two z-layers).
        num_layers (int): Total number of Fe layers in the final stack.
        vacuum (float): Vacuum spacing in Å to add around the final stack.
        output_path (str): If given, write the final structure to this path.

    Returns:
        Atoms: Final structure with multiple Fe layers.
    """
    __version__ = "0.0.1"
    
    if num_layers < 1:
        raise ValueError("Number of layers must be at least 1.")

    # Separate the single Fe slab into top and bottom layers
    fe_z_positions = slab_fe.get_positions()[:, 2]
    z_sorted = np.sort(np.unique(fe_z_positions))
    if len(z_sorted) < 2:
         raise ValueError("Input slab_fe must have at least two distinct z-positions for top and bottom layers.")

    z_bottom_single = z_sorted[0]
    z_top_single = z_sorted[1]
    layer_spacing = z_top_single - z_bottom_single

    slab_fe_bottom = slab_fe[[atom.position[2] == z_bottom_single for atom in slab_fe]].copy()
    slab_fe_top = slab_fe[[atom.position[2] == z_top_single for atom in slab_fe]].copy()

    # Start with the first layer based on the pattern
    if num_layers == 1:
        fe_stack = slab_fe_top.copy()
    elif num_layers == 2:
        fe_stack = slab_fe_bottom.copy()
        new_top_layer = slab_fe_top.copy()
        current_top_z = fe_stack.get_positions()[:, 2].max()
        dz_top = current_top_z - new_top_layer.get_positions()[:, 2].min() + layer_spacing
        new_top_layer.translate((0, 0, dz_top))
        fe_stack.extend(new_top_layer)
    else:
        # For 3 or more layers, build in a zig-zag pattern
        fe_stack = slab_fe.copy()
        current_top_z = fe_stack.get_positions()[:, 2].max()
        current_bottom_z = fe_stack.get_positions()[:, 2].min()
        print(f"{current_top_z=}")
        print(f"{current_bottom_z=}")
        print(f"{current_top_z - current_bottom_z=}")

        for i in range(1, num_layers - 1):
            if i % 2 != 0: # Add top layer
                new_bottom_layer = slab_fe_top.copy()
                dz_top = current_bottom_z - new_bottom_layer.get_positions()[:, 2].max() - layer_spacing
                new_bottom_layer.translate((0, 0, dz_top))
                fe_stack.extend(new_bottom_layer)
                current_bottom_z = fe_stack.get_positions()[:, 2].min()
            else: # Add bottom layer
                new_bottom_layer = slab_fe_bottom.copy()
                dz_bottom = current_bottom_z - new_bottom_layer.get_positions()[:, 2].max() - layer_spacing
                new_bottom_layer.translate((0, 0, dz_bottom))
                fe_stack.extend(new_bottom_layer)
                current_bottom_z = fe_stack.get_positions()[:, 2].min()


    # Re-center the slab along the z-axis with vacuum
    fe_stack.center(vacuum=vacuum, axis=2)

    # Write output if requested
    if output_path:
        write(output_path, fe_stack)

    return fe_stack