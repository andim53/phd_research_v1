import random 
from ase.build import surface, bulk
from ase.io import write

def remove_random_atoms_by_species(atoms, species, count):
    """
    Random. Specific number of atoms. Remove.
    """

    target_indices = [atom.index for atom in atoms if atom.symbol == species]
    if len(target_indices) < count:
        count = len(target_indices)
    
    indices_to_remove = random.sample(target_indices, count)

    new_atoms = atoms.copy()
    del new_atoms[indices_to_remove]

    return new_atoms

# fe_bulk = bulk('Fe', 'bcc', a=2.866, cubic=True)
# slab_fe_base = surface(fe_bulk, (0, 0, 1), layers=1, vacuum=10).repeat((5,5,1))
# write('slab_fe_base.xsf', slab_fe_base)

# cut_slab_fe_base = remove_random_atoms_by_species(slab_fe_base, 'Fe', 5)
# write('cut_slab_fe_base.xsf', cut_slab_fe_base)