import numpy as np

def generate_volume_variants(atoms, percent=1.0, n=5):
    vols = np.linspace(-percent, percent, n)
    variants = []

    for dv in vols:
        scale = (1.0 + dv / 100.0) ** (1/3)  # isotropic volume scaling
        a = atoms.copy()
        a.set_cell(atoms.cell * scale, scale_atoms=True)
        variants.append(a)

    return variants

# ta_new_cells = generate_volume_variants(ta_bulk, percent=1.0, n=5)
