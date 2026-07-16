from ase import Atoms
from ase.build import bulk, surface, add_vacuum
import numpy as np
from ase.io import write

from icecream import ic


def build_amorphous_TaB_layer(
    thickness,
    b_at_percent=20.0,
    cell_xy=(10.0, 10.0),
    min_dist=1.8,
    seed=0
):
    """
    Build an amorphous Ta–B layer.

    thickness: target thickness in Å (z direction)
    b_at_percent: B atomic percent (typically 15–30, default 20)
    cell_xy: (Lx, Ly) in-plane cell lengths (Å)
    min_dist: minimum allowed interatomic distance (Å)
    """
    rng = np.random.default_rng(seed)

    # Composition fractions
    f_B = b_at_percent / 100.0
    f_Ta = 1.0 - f_B

    # Atom density (rough, tunable)
    n_atoms_per_A3 = 0.06
    volume = cell_xy[0] * cell_xy[1] * thickness
    n_tot = int(volume * n_atoms_per_A3)

    n_B = int(round(f_B * n_tot))
    n_Ta = max(1, n_tot - n_B)

    symbols = ['Ta'] * n_Ta + ['B'] * n_B
    rng.shuffle(symbols)

    positions = []
    Lx, Ly = cell_xy
    Lz = thickness

    def ok_pos(pos, positions):
        for p in positions:
            d = pos - p
            d[0] -= Lx * np.rint(d[0] / Lx)
            d[1] -= Ly * np.rint(d[1] / Ly)
            if np.linalg.norm(d) < min_dist:
                return False
        return True

    while len(positions) < len(symbols):
        pos = np.array([
            rng.random() * Lx,
            rng.random() * Ly,
            rng.random() * Lz
        ])
        if ok_pos(pos, positions):
            positions.append(pos)

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_cell([Lx, Ly, Lz])
    atoms.set_pbc((True, True, False))
    return atoms


def build_Si_oxide_TaB_stack(
    si_repeats=(4, 4, 8),
    oxide_thickness=10.0,
    oxide_O_at_percent=66.7,  # MTJ-typical: 64–68 at.% O
    tab_thickness=5.0,
    b_at_percent=20.0,
    vacuum_top=15.0,
    seed=0
):
    """
    Build Si(100) / amorphous SiOx / amorphous Ta–B stack.
    """
    rng = np.random.default_rng(seed)

    # --- Si(100) substrate ---
    a_Si = 5.431

    # bulk_mgo = bulk('MgO', 'rocksalt', a=a_mgo, cubic=True)
    # slab_mgo = surface(bulk_mgo, (0, 0, 1), layers=1, vacuum=vacuum)
    si_bulk = bulk('Si', 'diamond', a=a_Si, cubic=True)
    si_slab = surface(si_bulk, (1, 0, 0), layers = si_repeats[2], vacuum=0)
    si_slab = si_slab.repeat((si_repeats[0], si_repeats[1], 1))
    # add_vacuum(si_slab, 0.0)
    

    cell = si_slab.get_cell()
    Lx, Ly = cell[0, 0], cell[1, 1]
    

    # --- Amorphous SiOx layer (MTJ-relevant) ---
    z_top_si = si_slab.get_positions()[:, 2].max()

    n_atoms_per_A3_ox = 0.08
    volume_ox = Lx * Ly * oxide_thickness
    n_ox = int(volume_ox * n_atoms_per_A3_ox)

    f_O = oxide_O_at_percent / 100.0
    f_Si = 1.0 - f_O

    n_O_ox = max(1, int(round(f_O * n_ox)))
    n_Si_ox = max(1, n_ox - n_O_ox)

    ox_symbols = ['Si'] * n_Si_ox + ['O'] * n_O_ox
    rng.shuffle(ox_symbols)

    ox_positions = []

    def ok_pos_ox(pos, positions):
        for p in positions:
            d = pos - p
            d[0] -= Lx * np.rint(d[0] / Lx)
            d[1] -= Ly * np.rint(d[1] / Ly)
            if np.linalg.norm(d) < 1.6:
                return False
        return True

    while len(ox_positions) < len(ox_symbols):
        pos = np.array([
            rng.random() * Lx,
            rng.random() * Ly,
            z_top_si + 0.5 + rng.random() * oxide_thickness
        ])
        if ok_pos_ox(pos, ox_positions):
            ox_positions.append(pos)

    oxide = Atoms(symbols=ox_symbols, positions=ox_positions)
    oxide.set_cell(cell)
    oxide.set_pbc((True, True, True))

    stack = si_slab + oxide

    # --- Amorphous Ta–B layer (unchanged) ---
    tab_layer = build_amorphous_TaB_layer(
        thickness=tab_thickness,
        b_at_percent=b_at_percent,
        cell_xy=(Lx, Ly),
        min_dist=1.8,
        seed=seed + 1
    )

    z_top_stack = stack.get_positions()[:, 2].max()
    tab_pos = tab_layer.get_positions()
    tab_pos[:, 2] += z_top_stack + 2.0 - tab_pos[:, 2].min()
    tab_layer.set_positions(tab_pos)
    

    tab_layer.set_cell(stack.get_cell())
    tab_layer.set_pbc((True, True, True))

    full_stack = stack + tab_layer

    # --- Vacuum ---
    z_max = full_stack.get_positions()[:, 2].max()
    cell = full_stack.get_cell()
    
    
    cell[2, 2] = max(cell[2, 2], z_max + vacuum_top)
    ic(cell)
    full_stack.set_cell(cell)
    
    full_stack.set_pbc((True, True, False))

    return full_stack



if __name__ == "__main__":
    stack = build_Si_oxide_TaB_stack(
        si_repeats=(2, 2, 1),
        oxide_thickness=1.0,
        tab_thickness=5.0,
        b_at_percent=20.0,  # Ta80B20
        vacuum_top=15.0,
        seed=42
    )

    write("Si100_SiO2_Ta80B20.xyz",stack)
    write("Si100_SiO2_Ta80B20.xsf",stack)
    # stack.write("Si100_SiO2_Ta80B20.traj")
    # stack.write("Si100_SiO2_Ta80B20.xyz")
    # stack.write("Si100_SiO2_Ta80B20.xsf", format="xsf")

##################################################################
from ase.io import write

import random
from ase.build import bulk
from icecream import ic

# build 3x3x3 Ta bulk
atoms = bulk('Ta', 'bcc', a=3.30, cubic=True) * (3, 3, 3)

# random B substitution
concentration = 0.05  # fraction of Ta to replace
n_sub = int(len(atoms) * concentration)
ic(n_sub)


indices = random.sample(range(len(atoms)), n_sub)
for i in indices:
    atoms[i].symbol = 'B'

write('Ta_B_random.xyz', atoms)

################################################################
from ase import Atoms
from ase.build import bulk, surface, add_vacuum
import numpy as np

def build_amorphous_WTaB_layer(thickness, x, cell_xy, min_dist=1.8, seed=0):
    """
    Build an amorphous W_{100-x}(Ta82 B18)_x layer.
    thickness: target thickness in Å (z direction)
    x: percentage (0–100) of Ta-B mixture (Ta82 B18) replacing W
    cell_xy: (Lx, Ly) in-plane cell lengths (Å)
    min_dist: minimum allowed interatomic distance (Å)
    """
    rng = np.random.default_rng(seed)

    # Composition fractions
    x_frac = x / 100.0
    f_W  = 1.0 - x_frac
    f_Ta = x_frac * 0.82
    f_B  = x_frac * 0.18

    # Total number of atoms – adjust for desired density
    # Here, choose a simple constant density by atoms per volume.
    # You can tune n_atoms_per_A3 based on more realistic mass density.
    n_atoms_per_A3 = 0.06  # rough guess; adjust as needed
    volume = cell_xy[0] * cell_xy[1] * thickness
    n_tot = int(volume * n_atoms_per_A3)

    n_W  = max(1, int(round(f_W  * n_tot)))
    n_Ta = max(0, int(round(f_Ta * n_tot)))
    n_B  = max(0, int(round(f_B  * n_tot)))
    # Adjust total to match
    n_tot_adjusted = n_W + n_Ta + n_B

    symbols = ['W'] * n_W + ['Ta'] * n_Ta + ['B'] * n_B
    rng.shuffle(symbols)

    positions = []
    Lx, Ly = cell_xy
    Lz = thickness

    def ok_pos(pos, positions):
        for p in positions:
            d = pos - p
            # minimum image in xy, free in z (no PBC in z inside this layer object)
            d[0] -= Lx * np.rint(d[0] / Lx)
            d[1] -= Ly * np.rint(d[1] / Ly)
            if np.linalg.norm(d) < min_dist:
                return False
        return True

    while len(positions) < len(symbols):
        pos = np.array([rng.random() * Lx,
                        rng.random() * Ly,
                        rng.random() * Lz])
        if ok_pos(pos, positions):
            positions.append(pos)

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_cell([Lx, Ly, Lz])
    atoms.set_pbc((True, True, False))
    return atoms


def build_Si_oxide_WTaB_stack(
    si_repeats=(4, 4, 8),
    oxide_thickness=10.0,
    wtab_thickness=5.0,
    x=30.0,
    vacuum_top=15.0,
    seed=0
):
    """
    Build Si(100) / thermally oxidized Si (simple oxide) / amorphous W_{100-x}(Ta82 B18)_x stack.

    si_repeats: repeats for Si(100) slab: (nx, ny, nlayers)
    oxide_thickness: target oxide thickness in Å (simple amorphous-ish O+Si mix)
    wtab_thickness: W-Ta-B layer thickness in Å
    x: composition parameter (0–100) in W_{100-x}(Ta82 B18)_x
    vacuum_top: vacuum thickness above W-Ta-B (Å)
    seed: random seed
    """
    rng = np.random.default_rng(seed)

    # 1. Build Si(100) substrate
    a_Si = 5.431
    si_bulk = bulk('Si', 'diamond', a=a_Si)
    si_slab = surface(si_bulk, (1, 0, 0), si_repeats[2])
    si_slab = si_slab.repeat((si_repeats[0], si_repeats[1], 1))
    add_vacuum(si_slab, 0.0)  # no top vacuum yet

    cell = si_slab.get_cell()
    Lx, Ly = cell[0, 0], cell[1, 1]

    # 2. Simple "thermally oxidized" silicon layer
    # Here: generate a mixed Si+O amorphous layer on top of Si to mimic SiO2
    # More realistic: pre-built amorphous SiO2 slab.
    bottom_positions = si_slab.get_positions()
    z_top_si = bottom_positions[:, 2].max()

    # Choose atom density for oxide
    n_atoms_per_A3_ox = 0.08  # rough; adjust
    volume_ox = Lx * Ly * oxide_thickness
    n_ox_total = int(volume_ox * n_atoms_per_A3_ox)
    # Rough SiO2 stoichiometry: 1/3 Si, 2/3 O
    n_Si_ox = max(1, int(round(n_ox_total / 3)))
    n_O_ox  = max(2, n_ox_total - n_Si_ox)

    ox_symbols = ['Si'] * n_Si_ox + ['O'] * n_O_ox
    rng.shuffle(ox_symbols)

    ox_positions = []
    Lz_ox = oxide_thickness

    def ok_pos_ox(pos, positions):
        for p in positions:
            d = pos - p
            d[0] -= Lx * np.rint(d[0] / Lx)
            d[1] -= Ly * np.rint(d[1] / Ly)
            if np.linalg.norm(d) < 1.6:
                return False
        return True

    while len(ox_positions) < len(ox_symbols):
        pos = np.array([
            rng.random() * Lx,
            rng.random() * Ly,
            z_top_si + 0.5 + rng.random() * Lz_ox  # slightly above top Si, then fill up
        ])
        if ok_pos_ox(pos, ox_positions):
            ox_positions.append(pos)

    oxide = Atoms(symbols=ox_symbols, positions=ox_positions)
    oxide.set_cell(cell)
    oxide.set_pbc((True, True, True))

    stack = si_slab + oxide

    # 3. Amorphous W-Ta-B layer
    wtab_layer = build_amorphous_WTaB_layer(
        thickness=wtab_thickness,
        x=x,
        cell_xy=(Lx, Ly),
        min_dist=1.8,
        seed=seed + 1
    )

    # Position W-Ta-B above oxide
    stack_pos = stack.get_positions()
    z_top_stack = stack_pos[:, 2].max()

    wtab_pos = wtab_layer.get_positions()
    z_min_wtab = wtab_pos[:, 2].min()
    shift = z_top_stack + 2.0 - z_min_wtab  # 2 Å gap
    wtab_pos[:, 2] += shift
    wtab_layer.set_positions(wtab_pos)

    # Unify cells and periodicity
    wtab_layer.set_cell(stack.get_cell())
    wtab_layer.set_pbc((True, True, True))

    full_stack = stack + wtab_layer

    # 4. Add vacuum on top
    # Extend cell in z to include vacuum_top above W-Ta-B
    positions = full_stack.get_positions()
    z_max = positions[:, 2].max()
    cell = full_stack.get_cell()
    z_old = cell[2, 2]
    z_new = max(z_old, z_max + vacuum_top)
    cell[2, 2] = z_new
    full_stack.set_cell(cell)

    return full_stack


if __name__ == "__main__":
    # Example usage:
    #   oxide_thickness = 10 Å
    #   W-Ta-B thickness = 5 Å
    #   x = 30 => W70(Ta82 B18)30
    stack = build_Si_oxide_WTaB_stack(
        si_repeats=(3, 3, 1),
        oxide_thickness=2.0,
        wtab_thickness=20.0,
        x=30.0,
        vacuum_top=15.0,
        seed=42
    )
    stack.write("Si100_thermOx_WTaB_x30_thick6A.traj")
    stack.write("Si100_thermOx_WTaB_x30_thick6A.xyz")
    stack.write("Si100_thermOx_WTaB_x30_thick6A.xsf", format="xsf")

##########################################################

from ase import Atoms
from ase.build import bulk, surface, add_vacuum
import numpy as np

def build_amorphous_WTaB_layer(thickness, x, cell_xy, min_dist=1.8, seed=0):
    """
    Build an amorphous W_{100-x}(Ta82 B18)_x layer.
    thickness: target thickness in Å (z direction)
    x: percentage (0–100) of Ta-B mixture (Ta82 B18) replacing W
    cell_xy: (Lx, Ly) in-plane cell lengths (Å)
    min_dist: minimum allowed interatomic distance (Å)
    """
    rng = np.random.default_rng(seed)

    # Composition fractions
    x_frac = x / 100.0
    f_W  = 1.0 - x_frac
    f_Ta = x_frac * 0.82
    f_B  = x_frac * 0.18

    # Total number of atoms – adjust for desired density
    # Here, choose a simple constant density by atoms per volume.
    # You can tune n_atoms_per_A3 based on more realistic mass density.
    n_atoms_per_A3 = 0.06  # rough guess; adjust as needed
    volume = cell_xy[0] * cell_xy[1] * thickness
    n_tot = int(volume * n_atoms_per_A3)

    n_W  = max(1, int(round(f_W  * n_tot)))
    n_Ta = max(0, int(round(f_Ta * n_tot)))
    n_B  = max(0, int(round(f_B  * n_tot)))
    # Adjust total to match
    n_tot_adjusted = n_W + n_Ta + n_B
    print(f"{n_tot_adjusted=}")

    symbols = ['W'] * n_W + ['Ta'] * n_Ta + ['B'] * n_B
    rng.shuffle(symbols)

    positions = []
    Lx, Ly = cell_xy
    Lz = thickness

    def ok_pos(pos, positions):
        for p in positions:
            d = pos - p
            # minimum image in xy, free in z (no PBC in z inside this layer object)
            d[0] -= Lx * np.rint(d[0] / Lx)
            d[1] -= Ly * np.rint(d[1] / Ly)
            if np.linalg.norm(d) < min_dist:
                return False
        return True

    while len(positions) < len(symbols):
        pos = np.array([rng.random() * Lx,
                        rng.random() * Ly,
                        rng.random() * Lz])
        if ok_pos(pos, positions):
            positions.append(pos)

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_cell([Lx, Ly, Lz])
    atoms.set_pbc((True, True, False))
    return atoms


def build_Si_oxide_WTaB_stack(
    si_repeats=(3, 3, 2),  # Now repeats a cubic Si cell
    oxide_thickness=1.0,
    wtab_thickness=6.0,
    x=30.0,
    vacuum_top=15.0,
    seed=0
):
    rng = np.random.default_rng(seed)

    # 1. Build CUBIC Si bulk first, then (100) surface
    a_Si = 5.431
    si_bulk_cubic = bulk('Si', 'diamond', a=a_Si, cubic=True)  # <- KEY CHANGE: cubic=True
    #print(f"Cubic Si bulk cell lengths: {si_bulk_cubic.get_cell_lengths()}")
    
    si_slab = surface(si_bulk_cubic, (1, 0, 0), si_repeats[2])  # (100) surface
    si_slab = si_slab.repeat((si_repeats[0], si_repeats[1], 1))
    add_vacuum(si_slab, 0.0)

    cell = si_slab.get_cell()
    Lx, Ly = cell[0, 0], cell[1, 1]
    print(f"Si slab cell: Lx={Lx:.2f}, Ly={Ly:.2f}")

    # 2. Oxide layer (unchanged)
    bottom_positions = si_slab.get_positions()
    z_top_si = bottom_positions[:, 2].max()

    n_atoms_per_A3_ox = 0.08
    volume_ox = Lx * Ly * oxide_thickness
    n_ox_total = int(volume_ox * n_atoms_per_A3_ox)
    n_Si_ox = max(1, int(round(n_ox_total / 3)))
    n_O_ox  = max(2, n_ox_total - n_Si_ox)

    ox_symbols = ['Si'] * n_Si_ox + ['O'] * n_O_ox
    rng.shuffle(ox_symbols)

    ox_positions = []
    Lz_ox = oxide_thickness

    def ok_pos_ox(pos, positions):
        for p in positions:
            d = pos - p
            d[0] -= Lx * np.rint(d[0] / Lx)
            d[1] -= Ly * np.rint(d[1] / Ly)
            if np.linalg.norm(d) < 1.6:
                return False
        return True

    while len(ox_positions) < len(ox_symbols):
        pos = np.array([
            rng.random() * Lx,
            rng.random() * Ly,
            z_top_si + 0.5 + rng.random() * Lz_ox
        ])
        if ok_pos_ox(pos, ox_positions):
            ox_positions.append(pos)

    oxide = Atoms(symbols=ox_symbols, positions=ox_positions)
    oxide.set_cell(cell)
    oxide.set_pbc((True, True, True))

    stack = si_slab + oxide

    # 3. W-Ta-B layer (unchanged)
    wtab_layer = build_amorphous_WTaB_layer(
        thickness=wtab_thickness, x=x, cell_xy=(Lx, Ly), min_dist=1.8, seed=seed + 1
    )

    stack_pos = stack.get_positions()
    z_top_stack = stack_pos[:, 2].max()
    wtab_pos = wtab_layer.get_positions()
    z_min_wtab = wtab_pos[:, 2].min()
    shift = z_top_stack + 2.0 - z_min_wtab
    wtab_pos[:, 2] += shift
    wtab_layer.set_positions(wtab_pos)
    wtab_layer.set_cell(stack.get_cell())
    wtab_layer.set_pbc((True, True, True))

    full_stack = stack + wtab_layer

    # 4. Force FINAL CELL fully cubic (optional, after stacking)
    positions = full_stack.get_positions()
    z_max = positions[:, 2].max()
    a_cubic = Lx  # Use Si in-plane length
    cell = full_stack.get_cell()
    cell[2, 2] = max(cell[2, 2], z_max + vacuum_top)  # Ensure enough vacuum
    # Make exactly cubic
    cubic_cell = np.diag([a_cubic, a_cubic, a_cubic])
    full_stack.set_cell(cubic_cell, scale_atoms=False)  # No x,y rescaling

    #print(f"Final cubic cell: {full_stack.get_cell_lengths()}")
    return full_stack



if __name__ == "__main__":
    # Example usage:
    #   oxide_thickness = 10 Å
    #   W-Ta-B thickness = 5 Å
    #   x = 30 => W70(Ta82 B18)30
    stack = build_Si_oxide_WTaB_stack(
        si_repeats=(3, 3, 1),
        oxide_thickness=2.0,
        wtab_thickness=1.0,
        x=30.0,
        vacuum_top=15.0,
        seed=42
    )
    stack.write("Si100_thermOx_WTaB_x30_thick6A.traj")
    stack.write("Si100_thermOx_WTaB_x30_thick6A.xyz")

#####################################################

def build_amorphous_WTaB_bulk(a_cubic, x, n_atoms_per_A3=0.06, min_dist=1.8, seed=0):
    """
    Build a CUBIC bulk amorphous W_{100-x}(Ta82 B18)_x system.
    
    a_cubic: cubic cell length in Å (e.g., 15.0 for ~3000 Å³ volume)
    x: percentage (0–100) of Ta-B mixture (Ta82 B18) replacing W
    n_atoms_per_A3: target atoms per Å³ (tune for realistic density)
    min_dist: minimum allowed interatomic distance (Å)
    """
    rng = np.random.default_rng(seed)

    # Composition fractions
    x_frac = x / 100.0
    f_W  = 1.0 - x_frac
    f_Ta = x_frac * 0.82
    f_B  = x_frac * 0.18

    # Total number of atoms for cubic cell
    volume = a_cubic ** 3
    n_tot = int(volume * n_atoms_per_A3)
    
    n_W  = max(1, int(round(f_W  * n_tot)))
    n_Ta = max(0, int(round(f_Ta * n_tot)))
    n_B  = max(0, int(round(f_B  * n_tot)))
    n_tot_adjusted = n_W + n_Ta + n_B
    print(f"Volume: {volume:.1f} Å³, Target N: {n_tot}, Actual N: {n_tot_adjusted}")

    symbols = ['W'] * n_W + ['Ta'] * n_Ta + ['B'] * n_B
    rng.shuffle(symbols)

    positions = []
    L = a_cubic  # cubic length

    def ok_pos(pos, positions):
        """Check minimum distance with FULL 3D periodic boundary conditions"""
        for p in positions:
            d = pos - p
            # Minimum image convention in ALL 3 directions
            for i in range(3):
                d[i] -= L * np.rint(d[i] / L)
            if np.linalg.norm(d) < min_dist:
                return False
        return True

    # Pack atoms randomly with PBC in 3D
    attempts = 0
    max_attempts = n_tot_adjusted * 100  # safety limit
    
    while len(positions) < len(symbols) and attempts < max_attempts:
        pos = np.array([rng.random() * L,
                        rng.random() * L,
                        rng.random() * L])
        if ok_pos(pos, positions):
            positions.append(pos)
        attempts += 1
    
    if len(positions) < len(symbols):
        print(f"Warning: Only packed {len(positions)}/{len(symbols)} atoms after {attempts} attempts")
    
    atoms = Atoms(symbols=symbols[:len(positions)], positions=positions)  # trim if incomplete
    atoms.set_cell(np.diag([L, L, L]))  # cubic cell
    atoms.set_pbc((True, True, True))   # periodic in ALL directions
    return atoms


# Example usage:
if __name__ == "__main__":
    bulk_wtab = build_amorphous_WTaB_bulk(
        a_cubic=10.0,    # 15x15x15 Å cubic cell
        x=32.1,          # W_{67.9}(Ta82 B18)_{32.1} from your original paper
        n_atoms_per_A3=0.06,
        seed=42
    )
    # print(f"Final structure: {len(bulk_wtab)} atoms, cell {bulk_wtab.get_cell_lengths()}")
    bulk_wtab.write("amorphous_WTaB_bulk_cubic.xsf")
