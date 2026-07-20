import numpy as np
from ase import Atoms

# === Dynamic Hollow Site Detection & Filling Function
def add_adsorbate_to_hollows(atoms, symbol='B', height=0.0, num_atoms=1, seed=None):
    """
    Detects 4-fold hollow sites on the square-like top monolayer plane 
    and randomly populates them up to the capacity limit.
    """
    if seed is not None:
        np.random.seed(seed)
        
    positions = atoms.get_positions()
    z_coords = positions[:, 2]
    z_max = np.max(z_coords)
    
    # Isolate surface atoms within a tight Z window
    surface_indices = np.where(z_max - z_coords < 0.2)[0]
    surf_positions = positions[surface_indices]
    
    if len(surf_positions) < 4:
        raise ValueError("Not enough surface atoms detected to determine hollow locations.")

    xy_positions = surf_positions[:, :2]
    x_coords = np.unique(np.round(xy_positions[:, 0], 3))
    y_coords = np.unique(np.round(xy_positions[:, 1], 3))
    
    x_coords.sort()
    y_coords.sort()
    
    # Generate midpoints between grid atomic coordinates
    hollow_x = (x_coords[:-1] + x_coords[1:]) / 2
    hollow_y = (y_coords[:-1] + y_coords[1:]) / 2
    
    # Handle periodic cell boundary fallback extensions
    dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 2.87
    dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 2.87
    hollow_x = np.append(hollow_x, x_coords[-1] + dx / 2)
    hollow_y = np.append(hollow_y, y_coords[-1] + dy / 2)
    
    grid_x, grid_y = np.meshgrid(hollow_x, hollow_y)
    candidate_centers = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    cell = atoms.get_cell()
    max_x, max_y = cell[0, 0], cell[1, 1]
    
    valid_hollows = []
    for pt in candidate_centers:
        pt[0] = pt[0] % max_x
        pt[1] = pt[1] % max_y
        valid_hollows.append([pt[0], pt[1], z_max + height])
        
    valid_hollows = np.unique(np.round(np.array(valid_hollows), 3), axis=0)
    total_hollows = len(valid_hollows)
    
    print(f"--> Detected {total_hollows} total 4-fold hollow sites available on the {atoms.get_chemical_formula()} surface.")
    
    if num_atoms > total_hollows:
        print(f"    Warning: Requested {num_atoms} atoms but only {total_hollows} exist. Capping at capacity.")
        num_atoms = total_hollows

    chosen_indices = np.random.choice(total_hollows, size=num_atoms, replace=False)
    selected_hollows = valid_hollows[chosen_indices]
    
    output_atoms = atoms.copy()
    for pos in selected_hollows:
        output_atoms += Atoms(symbol, positions=[pos])
        
    return output_atoms