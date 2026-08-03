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
# Define the specific representative energy points you want to pick out
target_energies = [0.0, 0.01, 0.05, 0.001, 0.005]
# target_energies = [0.0, 0.0001, 0.0005, 0.0001, 0.00005]
tolerance = 0.0005
# tolerance = 0.000001  # Maximum allowed delta distance to capture a target match

dir_out = '0_analy'               # Directory for the analysis
dir_xsf_traj = '0_xsf_traj'       # Stores entire trajectory
trajectory_file = f"../{dir_out}/{dir_xsf_traj}/traj_61.traj"

# Peak Broadening Parameters for the continuous profile representation
two_theta_grid = np.linspace(30, 50, 1000)
fwhm = 0.40  # Slightly broadened peak width for clarity
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

# Initialize Pymatgen XRD calculator (Standard Cu K-alpha radiation)
xrd_calc = XRDCalculator(wavelength="CuKa")
adaptor = AseAtomsAdaptor()

# ==============================================================================
# CUSTOM PLOT STYLE (YOUR CONFIGURATION + EXTENDED PALETTE)
# ==============================================================================
# Extended to 6 distinct bold colors to make the extra data clearly distinguishable
colors = [
    '#FF0000',  # Pure Bold Red
    '#00FF00',  # Pure Bold Green
    '#0000FF',  # Pure Bold Blue
    '#00FFFF',  # Pure Bold Cyan
    '#FF00FF',  # Pure Bold Magenta
    '#FF7F00',  # Pure Bold Orange
]

custom_rc_params = {
    # Typography
    'font.size': 12,
    'font.family': 'serif',

    # Axes and Borders
    'axes.linewidth': 1.5,
    'axes.edgecolor': 'black',
    'axes.spines.top': True,
    'axes.spines.right': True,

    # Major Ticks (Outer edge "boxed" look)
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.top': True,
    'ytick.right': True,

    # Minor Ticks
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,

    # Miscellaneous (Grid and Legend)
    'axes.grid': False,
    'legend.frameon': True,
}

# Apply style update
plt.rcParams.update(custom_rc_params)

# Label formatting choice
e_label = r'$E_{i}-E_{glob}$ (eV/atom)'

# ==============================================================================
# 1. DATA INGESTION & ENERGY EVALUATION
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

# ==============================================================================
# 2. TARGETED ONE-STRUCTURE SELECTION (WITH DEDUPLICATION)
# ==============================================================================
datasets = []
seen_indices = set()  # Track unique index selections to avoid drawing duplicates over each other

for target in target_energies:
    abs_diffs = np.abs(relative_energies_per_atom - target)
    best_match_idx = np.argmin(abs_diffs)
    
    if abs_diffs[best_match_idx] <= tolerance:
        # Avoid duplicating exactly the same structural curve if targets overlap
        if best_match_idx in seen_indices:
            continue
        seen_indices.add(best_match_idx)
        
        dE_atom = relative_energies_per_atom[best_match_idx]
        matched_atoms = all_structures[best_match_idx]
        
        syms = matched_atoms.get_chemical_symbols()
        ta_count = syms.count('Ta')
        b_count = syms.count('B')
        
        # label = f"Target {target} eV/atom ({e_label} = {dE_atom:.4f})"
        # datasets.append({"atoms": matched_atoms, "label": label, "dE_atom": dE_atom})
        label = f"{dE_atom:.4f} eV/atom"
        datasets.append({"atoms": matched_atoms, "label": label, "dE_atom": dE_atom})
    else:
        print(f"Warning: No structure found within tolerance window for target energy {target} eV/atom")

datasets = sorted(datasets, key=lambda x: x["dE_atom"])

# ==============================================================================
# 3. RUN PYMATGEN XRD & GENERATE PLOT
# ==============================================================================
fig, ax = plt.subplots(figsize=(5.5, 3.5))

for rank, data in enumerate(datasets):
    atoms = data["atoms"]
    pmg_structure = adaptor.get_structure(atoms)
    
    # scaled=False to avoid arbitrary 100 max value normalization
    pattern = xrd_calc.get_pattern(pmg_structure, scaled=False, two_theta_range=(30, 50))
    
    # Broaden peaks continuously
    simulated_intensity = np.zeros_like(two_theta_grid)
    for peak_angle, intensity in zip(pattern.x, pattern.y):
        simulated_intensity += intensity * norm.pdf(two_theta_grid, peak_angle, sigma)
    
    # Pull individual distinct hex color from our extended list
    line_color = colors[rank % len(colors)]
    
    ax.plot(
        two_theta_grid, 
        simulated_intensity, 
        color=line_color, 
        lw=1.8, 
        label=data["label"],
        alpha=0.95
    )

# ==============================================================================
# 4. GRAPH LAYOUT & BOX LABELS
# ==============================================================================
ax.set_xlabel(r"$2\theta$ (degree)")
ax.set_ylabel("Absolute Structural Intensity (a.u.)")
# ax.set_title("Pymatgen Crystal XRD: Extended Custom Color Comparison", pad=15)
ax.set_xlim(30, 50)

# Render legend inside the framed box layout cleanly
ax.legend(loc="upper right", edgecolor='black')

output_img = "pymatgen_custom_styled_xrd.png"
plt.tight_layout()
plt.savefig(output_img, dpi=300)
print(f"\nSuccess! Plotted {len(datasets)} target structures using your layout configuration. Saved to '{output_img}'")
plt.show()