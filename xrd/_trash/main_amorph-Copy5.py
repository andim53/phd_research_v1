import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ase.io import read 
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from scipy.stats import norm

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
# Define the strict energy filter range in eV/atom
energy_min_per_atom = 0.05
energy_max_per_atom = 0.10

dir_out = '0_analy'               # Directory for the analysis
dir_xsf_traj = '0_xsf_traj'       # Stores entire trajectory
trajectory_file = f"../{dir_out}/{dir_xsf_traj}/traj_61.traj"

# Peak Broadening Parameters for the continuous profile representation
two_theta_grid = np.linspace(30, 50, 1000)
fwhm = 0.15  
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

# Initialize Pymatgen XRD calculator (Standard Cu K-alpha radiation)
xrd_calc = XRDCalculator(wavelength="CuKa")
adaptor = AseAtomsAdaptor()

# ==============================================================================
# 1. DATA INGESTION & PER-ATOM ENERGY FILTERING
# ==============================================================================
print(f"Reading structures from: {trajectory_file}")
all_structures = read(trajectory_file, index=":")

energies_per_atom = []
for atoms in all_structures:
    num_atoms = len(atoms)
    try:
        total_energy = atoms.get_potential_energy()
    except RuntimeError:
        total_energy = atoms.info.get('energy', None)
        if total_energy is None:
            raise RuntimeError("Atoms object has no potential energy calculated or stored.")
    energies_per_atom.append(total_energy / num_atoms)

energies_per_atom = np.array(energies_per_atom)
min_energy_per_atom = np.min(energies_per_atom)
relative_energies_per_atom = energies_per_atom - min_energy_per_atom

# --- UPDATED FILTER: Keep structures strictly between 0.05 and 0.10 eV/atom ---
valid_indices = np.where(
    (relative_energies_per_atom >= energy_min_per_atom) & 
    (relative_energies_per_atom <= energy_max_per_atom)
)[0]
print(f"Structures within {energy_min_per_atom} to {energy_max_per_atom} eV/atom window: {len(valid_indices)}")

# Build and sort datasets
datasets = []
for idx in valid_indices:
    dE_atom = relative_energies_per_atom[idx]
    
    syms = all_structures[idx].get_chemical_symbols()
    ta_count = syms.count('Ta')
    b_count = syms.count('B')
    
    label = f"Struct {idx} [Ta{ta_count}B{b_count}] (+{dE_atom:.5f} eV/atom)"
    datasets.append({"atoms": all_structures[idx], "label": label, "dE_atom": dE_atom})

# Sort ascending so lower energy structures in this bracket are drawn first
datasets = sorted(datasets, key=lambda x: x["dE_atom"])

# ==============================================================================
# 2. RUN PYMATGEN UN-SCALED COMPUTATIONS & PLOT PuBu GRADIENT
# ==============================================================================
fig, ax = plt.subplots(figsize=(10.5, 5.5))

# Use the Purple-Blue (PuBu) sequential colormap
cmap = plt.get_cmap("PuBu")

# --- UPDATED NORMALIZATION: Map colors smoothly between 0.05 and 0.10 ---
norm_scale = mcolors.PowerNorm(gamma=0.6, vmin=energy_min_per_atom, vmax=energy_max_per_atom)

for data in datasets:
    dE = data["dE_atom"]
    atoms = data["atoms"]
    
    # Convert to Pymatgen Structure object
    pmg_structure = adaptor.get_structure(atoms)
    
    # scaled=False turns off normalization to 100
    pattern = xrd_calc.get_pattern(pmg_structure, scaled=False, two_theta_range=(30, 50))
    
    # Generate continuous 2-theta profile using the calculated intensities
    simulated_intensity = np.zeros_like(two_theta_grid)
    for peak_angle, intensity in zip(pattern.x, pattern.y):
        simulated_intensity += intensity * norm.pdf(two_theta_grid, peak_angle, sigma)
    
    # Apply continuous color within the filtered range
    line_color = cmap(norm_scale(dE))
    line_width = 0.5
    line_alpha = 0.5  # Transparency helps handle heavily stacked line regions
        
    ax.plot(
        two_theta_grid, 
        simulated_intensity, 
        color=line_color, 
        lw=line_width, 
        alpha=line_alpha
    )

# ==============================================================================
# 3. COLORBAR & LAYOUT
# ==============================================================================
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_scale)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"$\Delta E$ from Global Minimum (eV / atom)", fontsize=11, labelpad=10)

ax.set_xlabel(r"$2\theta$ (degree)", fontsize=11)
ax.set_ylabel("Absolute Structural Intensity (a.u.)", fontsize=11)
ax.set_title(f"Pymatgen Crystal XRD: Energy Window {energy_min_per_atom} to {energy_max_per_atom} eV/atom", fontsize=12, pad=15)
ax.set_xlim(30, 50)

output_img = "pymatgen_pubu_filtered_gradient_xrd.png"
plt.tight_layout()
plt.savefig(output_img, dpi=300)
print(f"\nSuccess! Comparison graph containing {len(datasets)} structures saved to '{output_img}'")
plt.show()