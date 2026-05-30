import numpy as np

def calculate_best_so_far(energies):
    """Compute best-so-far (lowest) relative energies."""
    min_energy = np.nanmin(energies)
    rel_energies = [e - min_energy if not np.isnan(e) else np.nan for e in energies]
    best_so_far = None
    best_energies = []
    best_idx = []
    for idx, e in enumerate(rel_energies):
        if best_so_far is None or (not np.isnan(e) and e < best_so_far):
            best_so_far = e
            best_so_far_idx = idx
        best_energies.append(best_so_far)
        best_idx.append(idx)
        
    return best_energies, best_idx