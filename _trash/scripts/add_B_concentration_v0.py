import numpy as np
from ase import Atoms

def add_B_concentration(ta, cB, fracs=None, seed=None, make_structure=False):
    """
    cB: atomic fraction of B (e.g. 0.01 = 1 at.% B)
    fracs: optional (N,3) fractional positions
    make_structure: if False, only compute numbers and positions
    """
    rng = np.random.default_rng(seed)

    n_ta = len(ta)
    n_b = int(round(cB * n_ta / (1.0 - cB)))

    if fracs is None:
        fracs = rng.random((n_b, 3))
    else:
        fracs = np.asarray(fracs)
        assert fracs.shape == (n_b, 3)

    positions = fracs @ ta.cell

    if not make_structure:
        return n_b, n_ta

    ta = ta.copy()
    ta += Atoms('B' * n_b, positions=positions)
    return ta, n_b, n_ta