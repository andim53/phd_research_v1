import numpy as np
from ase.build import bulk
from ase.eos import EquationOfState
from gpaw import GPAW, FermiDirac
import os

# 1. Define the setup parameters
a_guess = 3.30  # Initial guess for BCC Tantalum (Å)
kpts = (6, 6, 6)

volumes = []
energies = []

dir_save = 'opt_ta'
os.makedirs(dir_save, exist_ok=True)

print("Starting volume scan using LCAO mode...")
for scaling in np.linspace(0.95, 1.05, 5):
    a_current = a_guess * scaling
    
    # Create the cubic BCC bulk structure (2 atoms in the cell)
    ta_bulk = bulk("Ta", "bcc", a=a_current, cubic=True)
    
    # Attach the GPAW calculator with your parameters
    calc = GPAW(
        mode={"name": "lcao"},
        basis="dzp",
        xc="PBE",
        kpts=kpts,
        symmetry="off",
        nbands="nao",
        mixer={"backend": "pulay", "beta": 0.05, "nmaxold": 5, "weight": 100},
        # convergence={"energy": 1e-4, "density": 1e-3, "eigenstates": 1e-3},
        occupations=FermiDirac(width=0.05),
        # maxiter=100,
        txt=f"{dir_save}/ta_{a_current:.2f}.txt",  # Separates logs for each step
    )
    ta_bulk.calc = calc
    
    # Calculate total energy and volume
    energy = ta_bulk.get_potential_energy()
    volume = ta_bulk.get_volume()
    
    volumes.append(volume)
    energies.append(energy)
    print(f"a = {a_current:.3f} Å | Volume = {volume:.2f} Å³ | Energy = {energy:.4f} eV")

# 2. Fit the Equation of State (EOS) to find the optimum
eos = EquationOfState(volumes, energies, eos='sj')  # Standard Birch-Murnaghan
v_opt, e_opt, B = eos.fit()

# 3. Calculate the optimized lattice constant 'a'
# For cubic=True, Volume = a^3  =>  a = Volume^(1/3)
a_opt = v_opt ** (1 / 3)
bulk_modulus_gpa = B * 160.217

# 4. Save results to a text file
output_filename = "opt_ta.txt"
with open(output_filename, "w") as f:
    f.write("=== Tantalum Bulk Lattice Optimization Summary ===\n")
    f.write(f"Initial guess 'a': {a_guess:.3f} Å\n\n")
    f.write("Scan Data:\n")
    f.write("Lattice Constant (Å) | Volume (Å³) | Energy (eV)\n")
    f.write("---------------------------------------------------\n")
    
    # Generate the scanned 'a' array again to match lengths properly
    a_scanned = np.linspace(0.95, 1.05, 5) * a_guess
    for a_curr, vol, eng in zip(a_scanned, volumes, energies):
        f.write(f"{a_curr:20.4f} | {vol:11.4f} | {eng:11.6f}\n")
        
    f.write("---------------------------------------------------\n\n")
    f.write("Optimization Results (Birch-Murnaghan Fit):\n")
    f.write(f"Optimized Bulk Volume:           {v_opt:.6f} Å³\n")
    f.write(f"Optimized Lattice Constant (a):  {a_opt:.6f} Å\n")
    f.write(f"Minimum Energy:                  {e_opt:.6f} eV\n")
    f.write(f"Bulk Modulus:                    {bulk_modulus_gpa:.2f} GPa\n")

print(f"\nOptimization complete. Results successfully saved to: {output_filename}")