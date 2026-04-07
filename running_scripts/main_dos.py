import matplotlib.pyplot as plt
import numpy as np
from gpaw import GPAW, FermiDirac
from ase.dft.dos import DOS
from ase.io import read

def calculate_and_compare_dos(structures, kpts=(4, 4, 1), seeds=None, savename="does_results.csv"):
    if seeds is None:
        seeds = [i for i in range(len(structures))]
    
    # Format: { 'energy_seed_101': [...], 'dos_seed_101': [...], ... }
    all_data = {}

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

        dos_obj = DOS(calc, npts=800, width=0.1)
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

    
    df = pd.DataFrame(results_storage)
    df.to_csv(savename, index=False)
    print(f"saved: {savename}")

    return results_storage
    
structs = read(f"traj_19.traj", index=':')
results = calculate_and_compare_dos(structs, kpts=(4, 4, 1))