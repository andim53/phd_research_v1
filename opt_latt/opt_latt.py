import os
import numpy as np
from ase.build import bulk
from ase.eos import EquationOfState
from gpaw import GPAW, FermiDirac

from scripts.generate_interstitial_alloy import generate_interstitial_alloy

# --- Configuration & Parameters ---
# 1. Structural Parameters
HOST_ELEMENT = "Pt"
INTERSTITIAL_ELEMENT = "P"
LATTICE_TYPE = "fcc"
SUPERCELL = (1, 1, 1)
NUM_TO_REMOVE = 3

# 2. Lattice Optimization Settings
A_GUESS = 3.92 * 1.02  # Initial guess with 2% increase (Å)
SCALING_FACTORS = np.linspace(0.95, 1.05, 5)

# 3. GPAW Calculator Settings
KPTS = (6, 6, 6)
CALC_MODE = {"name": "lcao"}
BASIS_SET = "dzp"
XC_FUNCTIONAL = "PBE"
FERMI_WIDTH = 0.05

# 4. Output Directories & File Paths
DIR_SAVE = 'opt_pt'
os.makedirs(DIR_SAVE, exist_ok=True)

# File paths are constructed to save directly inside DIR_SAVE
OUTPUT_FILENAME = os.path.join(DIR_SAVE, "opt_pt.txt")


# --- Initialization ---
volumes = []
energies = []


# --- 1. Volume Scan Loop (GPAW LCAO Calculations) ---
print("Starting volume scan using LCAO mode...")

for scaling in SCALING_FACTORS:
    a_current = A_GUESS * scaling
    
    # Structure setup
    pt_bulk = bulk(HOST_ELEMENT, LATTICE_TYPE, a=a_current, cubic=True)
    alloy_pt = generate_interstitial_alloy(
        host_unit=pt_bulk,
        interstitial_element=INTERSTITIAL_ELEMENT,
        supercell_dim=SUPERCELL,
        lattice_type=LATTICE_TYPE,
        num_to_remove=NUM_TO_REMOVE
    )
    
    # Calculator configuration (logs saved dynamically inside DIR_SAVE)
    gpaw_log_path = os.path.join(DIR_SAVE, f"pt_scale_{scaling:.2f}.txt")
    calc = GPAW(
        mode=CALC_MODE,
        basis=BASIS_SET,
        xc=XC_FUNCTIONAL,
        kpts=KPTS,
        symmetry="off",
        nbands="nao",
        mixer={"backend": "pulay", "beta": 0.05, "nmaxold": 5, "weight": 100},
        occupations=FermiDirac(width=FERMI_WIDTH),
        txt=gpaw_log_path, 
    )
    alloy_pt.calc = calc
    
    # Evaluation
    energy = alloy_pt.get_potential_energy()
    volume = alloy_pt.get_volume()
    
    volumes.append(volume)
    energies.append(energy)
    
    # Print progress with increment percentage
    pct_increment = (scaling - 1.0) * 100
    print(f"a = {a_current:.3f} Å ({pct_increment:+.1f}%) | Volume = {volume:.2f} Å³ | Energy = {energy:.4f} eV")


# --- 2. Equation of State Fitting ---
eos = EquationOfState(volumes, energies, eos='sj')  
v_opt, e_opt, B = eos.fit()

# Calculate optimized lattice metrics
a_opt = v_opt ** (1 / 3)          # Cubic relation: V = a^3
bulk_modulus_gpa = B * 160.217    # Convert eV/Å³ to GPa

# Calculate the final optimal increment percentage relative to initial guess
opt_pct_change = ((a_opt - A_GUESS) / A_GUESS) * 100


# --- 3. Data Logging & Output Generation ---
with open(OUTPUT_FILENAME, "w") as f:
    f.write("=== Platinum Bulk Lattice Optimization Summary ===\n")
    f.write(f"Initial guess 'a': {A_GUESS:.3f} Å\n\n")
    
    f.write("Scan Data:\n")
    f.write("Lattice Constant (Å) | Cell Increment (%) | Volume (Å³) | Energy (eV)\n")
    f.write("-----------------------------------------------------------------------\n")
    
    a_scanned = SCALING_FACTORS * A_GUESS
    for a_curr, scale, vol, eng in zip(a_scanned, SCALING_FACTORS, volumes, energies):
        pct_inc = (scale - 1.0) * 100
        f.write(f"{a_curr:20.4f} | {pct_inc:+18.2f}% | {vol:11.4f} | {eng:11.6f}\n")
        
    f.write("-----------------------------------------------------------------------\n\n")
    f.write("Optimization Results (Birch-Murnaghan Fit):\n")
    f.write(f"Optimized Bulk Volume:           {v_opt:.6f} Å³\n")
    f.write(f"Optimized Lattice Constant (a):  {a_opt:.6f} Å\n")
    f.write(f"Optimized Cell Increment:        {opt_pct_change:+.4f}% (relative to initial guess)\n")
    f.write(f"Minimum Energy:                  {e_opt:.6f} eV\n")
    f.write(f"Bulk Modulus:                    {bulk_modulus_gpa:.2f} GPa\n")

print(f"\nOptimization complete. Summary successfully saved to: {OUTPUT_FILENAME}")