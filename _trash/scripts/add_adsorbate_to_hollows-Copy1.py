import numpy as np
from ase import Atoms
from ase.visualize import view

def add_adsorbate_to_hollows(
    atoms: Atoms, 
    symbol: str, 
    height: float, 
    num_atoms: int = 1,
    seed: int = None,
):
    rng = np.random.default_rng(seed)
    fe_indices = np.array([a.index for a in atoms if a.symbol == 'Fe'])

    rng.shuffle(fe_indices)

    all_possible_sites = [fe_indices[i:i + 4] for i in range(0, len(fe_indices), 4)]
    all_possible_sites = [site for site in all_possible_sites if len(site) == 4]

    if not all_possible_sites:
        raise ValueError("Not enough Fe atoms to form even one quartet.")

    num_atoms = min(num_atoms, len(all_possible_sites))

    selected_sites = all_possible_sites[:num_atoms]

    new_positions = []
    for site in selected_sites:
        pos_targets = atoms.get_positions()[site]
        center_xy = np.mean(pos_targets[:, :2], axis=0)
        center_z = np.mean(pos_targets[:, 2]) + height
        new_positions.append([center_xy[0], center_xy[1], center_z])
        
    adsorbates = Atoms(symbol * len(new_positions), positions=new_positions)
    return atoms + adsorbates
