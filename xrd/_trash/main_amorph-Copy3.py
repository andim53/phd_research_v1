import numpy as np
import matplotlib.pyplot as plt
from ase.io import read 
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from scipy.stats import norm

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
# Custom energy filter cutoff in eV/atom (e.g., 0.001, 0.005, 0.01)
energy_threshold_per_atom = 0.1  

dir_out = '0_analy'               # Directory for the analysis
dir_xsf_traj = '0_xsf_traj'       # Stores entire trajectory
trajectory_file = f"../{dir_out}/{dir_xsf_traj}/traj_61.traj"

# ==============================================================================
# 1. DATA INGESTION AND PER-ATOM ENERGY FILTERING
# ==============================================================================
print(f"Reading structures from: {trajectory_file}")
all_structures = read(trajectory_file, index=":")

# Extract potential energies normalized per atom
energies_per_atom = []
for atoms in all_structures:
    num_atoms = len(atoms)
    try:
        total_energy = atoms.get_potential_energy()
    except RuntimeError:
        if 'energy' in atoms.info:
            total_energy = atoms.info['energy']
        else:
            raise RuntimeError("Atoms object has no potential energy calculated or stored.")
    
    # Store energy divided by total number of atoms in this specific structure
    energies_per_atom.append(total_energy / num_atoms)

energies_per_atom = np.array(energies_per_atom)
min_energy_per_atom = np.min(energies_per_atom)
relative_energies_per_atom = energies_per_atom - min_energy_per_atom

# Filter criteria dynamically adjusted using the per-atom threshold
valid_indices = np.where(relative_energies_per_atom <= energy_threshold_per_atom)[0]

print(f"Total structures in trajectory: {len(all_structures)}")
print(f"Structures within {energy_threshold_per_atom} eV/atom of minimum: {len(valid_indices)}")

# ==============================================================================
# 2. DYNAMICALLY BUILD THE DATASETS CONFIGURATION
# ==============================================================================
# Linestyle cycle for non-global-minima structures
linestyle_cycle = ["--", "-.", ":"]

datasets = []
for idx in valid_indices:
    dE_per_atom = relative_energies_per_atom[idx]
    
    # Label indicates trajectory index and normalized energy stability per atom
    label = f"Struct {idx} (+{dE_per_atom:.5f} eV/atom)"
    if dE_per_atom == 0.0:
        label += " [Global Min]"
        
    datasets.append({
        "atoms": all_structures[idx],
        "label": label,
        "dE_atom": dE_per_atom
    })

# Sort by energy per atom so Global Min (dE_atom=0.0) is ALWAYS first (at the back)
datasets = sorted(datasets, key=lambda x: x["dE_atom"])

# Assign linestyles after sorting
for rank, data in enumerate(datasets):
    if data["dE_atom"] == 0.0:
        data["linestyle"] = "-"  # Force Global Minimum to always be a solid line
    else:
        # Cycle through other line options for non-global minima
        data["linestyle"] = linestyle_cycle[(rank - 1) % len(linestyle_cycle)]

# ==============================================================================
# 3. XRD CONFIGURATION AND CALCULATION (FROM YOUR SETTINGS)
# ==============================================================================
two_theta_grid = np.linspace(30, 50, 1000)
fwhm = 0.1  
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

# Initialize XRD calculator (Standard Cu K-alpha radiation)
xrd_calc = XRDCalculator(wavelength="CuKa")
adaptor = AseAtomsAdaptor()

# Setup Plot
plt.figure(figsize=(9, 5.5))

# Loop through each filtered structure to process and plot
for data in datasets:
    atoms = data["atoms"]

    # Convert to a Pymatgen structure directly (No supercell expansion)
    pmg_structure = adaptor.get_structure(atoms)

    # Get XRD pattern details
    # pattern = xrd_calc.get_pattern(pmg_structure, two_theta_range=(30, 50))
    # Pass scaled=False to get absolute structural intensities
    pattern = xrd_calc.get_pattern(pmg_structure, scaled=False, two_theta_range=(30, 50))

    # Generate continuous 2-theta grid and apply peak broadening
    simulated_intensity = np.zeros_like(two_theta_grid)
    for peak_angle, intensity in zip(pattern.x, pattern.y):
        simulated_intensity += intensity * norm.pdf(two_theta_grid, peak_angle, sigma)

    # Normalize intensity to maximum 100
    if max(simulated_intensity) > 0:
        simulated_intensity = (simulated_intensity / max(simulated_intensity)) * 100

    # Add a slight background baseline so it doesn't drop to absolute 0
    simulated_intensity += 5 

    # Plot using thin black lines with the dynamically assigned style
    plt.plot(
        two_theta_grid, 
        simulated_intensity, 
        label=data["label"], 
        color="black", 
        linestyle=data["linestyle"], 
        lw=1.2
    )

# ==============================================================================
# 4. FINALIZE PLOT FORMATTING
# ==============================================================================
plt.xlabel(r"$2\theta$ (degree)", fontsize=11)
plt.ylabel("Intensity (arbitrary units)", fontsize=11)
plt.title(f"XRD Comparison: Formations Below {energy_threshold_per_atom} eV/atom Range", fontsize=13, pad=15)
# plt.legend(frameon=True, fontsize=10, loc="upper right")
plt.xlim(30, 50)
plt.grid(True, linestyle="--", alpha=0.3)

# Save and show
output_img = "ta_comparison_xrd.png"
plt.tight_layout()
plt.savefig(output_img, dpi=300)
print(f"Comparison graph containing {len(datasets)} structures saved as '{output_img}'")
plt.show()