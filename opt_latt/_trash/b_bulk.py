import numpy as np
from ase.io import read
from ase.eos import EquationOfState
from gpaw import GPAW, FermiDirac

import os

# 1. Load the Boron structure from your CIF file
# Make sure "B_bulk.cif" is in the same folder as this script
b_bulk = read("B_bulk.cif")

# Store the baseline initial cell and volume
initial_cell = b_bulk.get_cell()
initial_volume = b_bulk.get_volume()

# 2. Define the setup parameters
kpts = (6, 6, 6)

volumes = []
energies = []

dir_save = 'opt_b'
os.makedirs(dir_save,exist_ok=True)

print(f"Starting uniform volume scan for Boron (Initial Volume = {initial_volume:.2f} Å³)...")

# We will scale the cell scale-factor linearly from 0.95 to 1.05
# Note: Volume scales as scaling_factor^3
for scaling_factor in np.linspace(0.95, 1.05, 5):
    # Apply uniform scaling to the unit cell vectors
    new_cell = initial_cell * scaling_factor
    b_bulk.set_cell(new_cell, scale_atoms=True)  # scale_atoms=True scales internal coordinates
    
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
        txt=f"{dir_save}/b_scale_{scaling_factor:.2f}.txt",  # Separates logs for each step
    )
    b_bulk.calc = calc
    
    # Calculate total energy and volume
    energy = b_bulk.get_potential_energy()
    volume = b_bulk.get_volume()
    
    volumes.append(volume)
    energies.append(energy)
    print(f"Scale: {scaling_factor:.3f} | Volume = {volume:.2f} Å³ | Energy = {energy:.4f} eV")

# 3. Fit the Equation of State (EOS) to find the optimum volume
eos = EquationOfState(volumes, energies, eos='sj')  # Standard Birch-Murnaghan
v_opt, e_opt, B = eos.fit()

# 4. Calculate the optimal scaling factor for the cell parameters
# Since Volume is proportional to length^3:
opt_scaling_factor = (v_opt / initial_volume) ** (1 / 3)

# Reconstruct the optimized cell to get the final lattice parameters
opt_cell = initial_cell * opt_scaling_factor
b_bulk.set_cell(opt_cell, scale_atoms=True)
opt_lengths = b_bulk.cell.cellpar()[:3]   # [a, b, c]
opt_angles = b_bulk.cell.cellpar()[3:]    # [alpha, beta, gamma]

bulk_modulus_gpa = B * 160.217

# 5. Save results to a text file
output_filename = "boron_optimization_results.txt"
with open(output_filename, "w") as f:
    f.write("=== Boron Bulk Lattice Optimization Summary ===\n")
    f.write(f"Source file: B_bulk.cif\n")
    f.write(f"Initial cell volume: {initial_volume:.4f} Å³\n\n")
    f.write("Scan Data:\n")
    f.write("Scale Factor | Volume (Å³) | Energy (eV)\n")
    f.write("---------------------------------------------------\n")
    
    scaling_factors = np.linspace(0.95, 1.05, 5)
    for sf, vol, eng in zip(scaling_factors, volumes, energies):
        f.write(f"{sf:12.4f} | {vol:11.4f} | {eng:11.6f}\n")
        
    f.write("---------------------------------------------------\n\n")
    f.write("Optimization Results (Birch-Murnaghan Fit):\n")
    f.write(f"Optimized Bulk Volume:           {v_opt:.6f} Å³\n")
    f.write(f"Optimal Cell Scale Factor:       {opt_scaling_factor:.6f}\n")
    f.write(f"Minimum Energy:                  {e_opt:.6f} eV\n")
    f.write(f"Bulk Modulus:                    {bulk_modulus_gpa:.2f} GPa\n\n")
    
    f.write("Optimized Cell Parameters:\n")
    f.write(f"a: {opt_lengths[0]:.4f} Å, b: {opt_lengths[1]:.4f} Å, c: {opt_lengths[2]:.4f} Å\n")
    f.write(f"alpha: {opt_angles[0]:.2f}°, beta: {opt_angles[1]:.2f}°, gamma: {opt_angles[2]:.2f}°\n")

print(f"\nOptimization complete. Results successfully saved to: {output_filename}")