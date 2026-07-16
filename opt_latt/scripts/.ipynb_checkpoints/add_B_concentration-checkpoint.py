import numpy as np
from ase import Atoms
from scipy.spatial import KDTree

def add_B_concentration(ta, cB, interstitial_type='hollow', fracs=None, seed=None, make_structure=False):
    """
    cB: atomic fraction of B (e.g. 0.01 = 1 at.% B)
    interstitial_type: 'bridge' (between 2 atoms) or 'hollow' (between 4 atoms)
    fracs: optional (N,3) fractional positions (overrides automated interstitial generation)
    make_structure: if False, only compute numbers and positions
    """
    rng = np.random.default_rng(seed)

    n_ta = len(ta)
    n_b = int(round(cB * n_ta / (1.0 - cB)))

    if fracs is None:
        # 1. Get Cartesian coordinates of the substrate matrix
        ta_pos = ta.get_positions()
        cell_diag = np.linalg.norm(ta.cell.diagonal())
        
        # 2. Use a KDTree to find nearest neighbors considering periodic boundaries
        # We estimate a search radius slightly larger than typical nearest-neighbor spacing
        # For typical lattices, nearest neighbor is ~2.5-3.0 Å. Adjust if your system differs.
        tree = KDTree(ta_pos, boxsize=ta.cell.diagonal())
        
        valid_sites = []
        
        if interstitial_type == 'bridge':
            # Find pairs: search for neighbors within a typical bond distance window
            # Adjust the radius (e.g., 3.0) to match your specific substrate's lattice constant
            pairs = tree.query_pairs(r=3.0)
            for i, j in pairs:
                # Midpoint between 2 atoms (accounting for periodic boundary wrapping via cell)
                diff = ta_pos[j] - ta_pos[i]
                # Wrap displacement vector to nearest image
                diff = diff - np.round(diff / ta.cell.diagonal()) * ta.cell.diagonal()
                midpoint = ta_pos[i] + 0.5 * diff
                valid_sites.append(midpoint)
                
        elif interstitial_type == 'hollow':
            # Find groups of 4 neighboring atoms forming a square/plaquette
            # We query the nearest neighbors for every single atom
            for i in range(n_ta):
                # Query nearest neighbors (itself + 4 neighbors = 5)
                distances, indices = tree.query(ta_pos[i], k=5)
                # Exclude itself (the first index)
                neighbors = indices[1:] 
                
                # A simple hollow site estimate is the average of an atom and its local 2D shell neighbors
                # We handle periodic boundary wrapping relative to the reference atom `i`
                cluster = ta_pos[neighbors] - ta_pos[i]
                cluster = cluster - np.round(cluster / ta.cell.diagonal()) * ta.cell.diagonal()
                center = ta_pos[i] + np.mean(cluster, axis=0)
                valid_sites.append(center)
        else:
            raise ValueError("interstitial_type must be 'bridge' or 'hollow'")

        # Remove any coordinate duplicates that arise from overlapping neighbor definitions
        valid_sites = np.unique(np.round(valid_sites, decimals=4), axis=0)

        if len(valid_sites) < n_b:
            raise ValueError(f"Requested n_b ({n_b}) exceeds available {interstitial_type} sites ({len(valid_sites)}).")

        # 3. Randomly select n_b positions from the high-symmetry site pool
        chosen_indices = rng.choice(len(valid_sites), size=n_b, replace=False)
        positions = valid_sites[chosen_indices]
        
        # Convert back to fractional coordinates for standard code tracking
        # Using pseudo-inverse to handle non-orthogonal cells if necessary
        fracs = positions @ np.linalg.pinv(ta.cell)
        
    else:
        fracs = np.asarray(fracs)
        assert fracs.shape == (n_b, 3)
        positions = fracs @ ta.cell

    if not make_structure:
        return n_b, n_ta

    ta_copied = ta.copy()
    ta_copied += Atoms('B' * n_b, positions=positions)
    
    # Wrap atoms back cleanly inside the unit cell boundaries
    ta_copied.wrap()
    
    return ta_copied, n_b, n_ta