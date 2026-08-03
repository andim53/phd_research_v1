import numpy as np
import matplotlib.pyplot as plt
from ase.io import read 
from scipy.stats import norm

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
energy_threshold_per_atom = 0.05   # Adjust to include both of your target configurations

dir_out = '0_analy'               # Directory for the analysis
dir_xsf_traj = '0_xsf_traj'       # Stores entire trajectory
trajectory_file = f"../{dir_out}/{dir_xsf_traj}/traj_43.traj"

# X-ray Source Setup (Cu K-alpha: wavelength = 1.54184 Å)
wavelength = 1.54184
two_theta_grid = np.linspace(30, 50, 500)  # 2-theta range

# Atomic form factor approximations f(q) weighted by atomic numbers
Z_factors = {'Ta': 73.0, 'B': 5.0}

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

valid_indices = np.where(relative_energies_per_atom <= energy_threshold_per_atom)[0]
print(f"Structures within {energy_threshold_per_atom} eV/atom window: {len(valid_indices)}")

# Build and sort datasets so Global Min is plotted first (at the back)
datasets = []
for idx in valid_indices:
    dE_atom = relative_energies_per_atom[idx]
    
    syms = all_structures[idx].get_chemical_symbols()
    ta_count = syms.count('Ta')
    b_count = syms.count('B')
    
    label = f"Struct {idx} [Ta{ta_count}B{b_count}] (+{dE_atom:.5f} eV/atom)"
    if dE_atom == 0.0:
        label += " [Global Min]"
        
    datasets.append({"atoms": all_structures[idx], "label": label, "dE_atom": dE_atom})

# Sort ascending: Global Min (0.0) is at index 0 (rendered first/at the very back)
datasets = sorted(datasets, key=lambda x: x["dE_atom"])

# ==============================================================================
# 2. VARIABLE-STOICHIOMETRY DEBYE SCATTERING SOLVER
# ==============================================================================
def calculate_debye_intensity_variable_stoich(atoms, theta_range, wl):
    """Calculates normalized scattering intensity using the Debye formula
    adjusted specifically for variable structural compositions.
    """
    theta_rad = np.radians(theta_range / 2.0)
    q_grid = (4.0 * np.pi * np.sin(theta_rad)) / wl
    
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    f_factors = np.array([Z_factors.get(sym, 1.0) for sym in symbols])
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    
    intensity_profile = np.zeros_like(q_grid)
    normalization_factor = np.sum(f_factors**2)
    
    for idx_q, q in enumerate(q_grid):
        if q == 0:
            intensity_profile[idx_q] = (np.sum(f_factors)**2) / normalization_factor
            continue
            
        qr = q * r_matrix
        np.fill_diagonal(qr, 1.0) 
        sin_qr_over_qr = np.sin(qr) / qr
        np.fill_diagonal(sin_qr_over_qr, 1.0) 
        
        f_cross = f_factors[:, np.newaxis] * f_factors[np.newaxis, :]
        total_q_intensity = np.sum(f_cross * sin_qr_over_qr)
        
        intensity_profile[idx_q] = total_q_intensity / normalization_factor
        
    return intensity_profile

# ==============================================================================
# 3. RUN COMPUTATIONS & GENERATE GRADIENT PLOT
# ==============================================================================
fig, ax = plt.subplots(figsize=(10.5, 5.5))

# Setup color gradient for non-minimum landscape elements
cmap = plt.get_cmap("viridis_r")

for data in datasets:
    dE = data["dE_atom"]
    atoms = data["atoms"]
    intensity = calculate_debye_intensity_variable_stoich(atoms, two_theta_grid, wavelength)
    
    if dE == 0.0:
        # Standout styling for absolute baseline global min structure at the back
        line_color = "black"
        line_width = 2.2
        line_style = "-"
        line_alpha = 1.0
    else:
        # Map color continuously relative to your upper filter window boundary
        line_color = cmap(dE / energy_threshold_per_atom)
        line_width = 1.0
        line_style = "-"
        line_alpha = 0.6  # Transparency cuts down on visual overlapping issues
        
    ax.plot(
        two_theta_grid, 
        intensity, 
        color=line_color, 
        linestyle=line_style, 
        lw=line_width,
        alpha=line_alpha
    )

# ==============================================================================
# 4. COLORBAR & COMPACT GRAPH LAYOUT
# ==============================================================================
# Add continuous Color bar representation on the right
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=energy_threshold_per_atom))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"$\Delta E$ from Global Minimum (eV / atom)", fontsize=11, labelpad=10)

ax.set_xlabel(r"$2\theta$ (degree)", fontsize=11)
ax.set_ylabel("Normalized Scattering Intensity $I(q) / \sum f_i^2$", fontsize=11)
ax.set_title("Debye Scattering: Energy Landscape Structural Profiles", fontsize=12, pad=15)
ax.set_xlim(30, 50)

output_img = "debye_variable_stoich_comparison.png"
plt.tight_layout()
plt.savefig(output_img, dpi=300)
print(f"\nSuccess! Comparison graph containing {len(datasets)} structures saved to '{output_img}'")
plt.show()