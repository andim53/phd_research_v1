import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gpaw import GPAW, FermiDirac
from ase.io import read


def calculate_and_compare_dos(structures, kpts=(12, 12, 1), seeds=None, savename="dos_results_high_res.csv"):
    if seeds is None:
        seeds = [i for i in range(len(structures))]
    
    # Format: { 'energy_seed_101': [...], 'dos_seed_101': [...], ... }
    results_storage = {}

    for atoms, seed in zip(structures, seeds):
        print(f"run seed: {seed}")
    
        calc = GPAW(
            mode={"name": "lcao"},
            basis="dzp",
            xc="PBE",
            mixer={"backend": "pulay", "beta": 0.05, "nmaxold": 5, "weight": 100},
            convergence={"energy": 1e-4, "density": 1e-3, "eigenstates": 1e-3},
            txt=f"output_seed_{seed}.txt",
            kpts=kpts,
            symmetry='off',
            nbands='nao',
            maxiter=300,
            occupations={"name": "fermi-dirac", "width": 0.05},
            hund=True,
            spinpol=True
        )

        atoms.calc = calc
        atoms.get_potential_energy()
        e_fermi = calc.get_fermi_level()

        npts = 2000     # Higher resolution for the energy axis
        width = 0.15    # Smearing width (Gaussian)

        # calc dos
        dos_obj = DOS(calc, npts=npts, width=width) #try 0.3 
        energies = dos_obj.get_energies() - e_fermi  

        dos_up = np.zeros_like(energies)
        dos_down = np.zeros_like(energies)
        total_dos = np.zeros_like(energies)

        if calc.get_number_of_spins() == 2:
            dos_up = dos_obj.get_dos(spin=0)
            dos_down = dos_obj.get_dos(spin=1)
            total_dos = dos_up + dos_down
        else:
            total_dos = dos_obj.get_dos()
            dos_up = total_dos
        
        results_storage[f'energy_{seed}'] = energies
        results_storage[f'total_dos_{seed}'] = total_dos
        results_storage[f'spin_up_{seed}'] = dos_up
        results_storage[f'spin_down_{seed}'] = dos_down

        # calc partial dos
        symbols = atoms.get_chemical_symbols()

        for i, symbol in enumerate(symbols):
            if symbol in ['Fe', 'Ni', 'Co', 'Mo', 'Cr']:
                # extract l=2 (d-orbitals), m=0 (dz2)
                # _, dz2_up = calc.get_orbital_ldos(a=i, spin=0, angular='d', npts=npts, width=width)
                # Note: GPAW returns [energies, sum_of_all_5_d_orbitals] by default.
                # To get ONLY dz2, we often need to use the 'm=0' projection:
                _, dz2_up = calc.get_all_electron_ldos(atoms=[i], spin=0, angular='d', m=0, npts=npts, width=width)
                _, dz2_down = calc.get_all_electron_ldos(atoms=[i], spin=1, angular='d', m=0, npts=npts, width=width)

                results_storage[f'dz2_atom{i}_{symbol}_up_{seed}'] = dz2_up
                results_storage[f'dz2_atom{i}_{symbol}_down_{seed}'] = dz2_down
    
    df = pd.DataFrame(results_storage)
    df.to_csv(savename, index=False)
    print(f"saved: {savename}")

    return results_storage
    
structs = read(f"ref_island.traj", index=':')
results = calculate_and_compare_dos(structs, kpts=(12, 12, 1))
