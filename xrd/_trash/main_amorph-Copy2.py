import numpy as np
import matplotlib.pyplot as plt
from ase.io import read 
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from scipy.stats import norm

# ==========================================
# CONFIGURATION OPTIONS: MULTIPLE FILES
# ==========================================
# All files set to black with differing linestyles for monochromatic clarity
datasets = [
    {"file": "Ta0b struct_700.xsf", "label": r"Ta Bulk (x=0)", "linestyle": "-"},  # Solid
    {"file": "Ta1b struct_489.xsf", "label": r"Ta1B",          "linestyle": "--"}, # Dashed
    {"file": "Ta3b struct_489.xsf", "label": r"Ta3B",          "linestyle": ":"}   # Dotted
]

two_theta_grid = np.linspace(30, 50, 1000)
fwhm = 0.1  
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

# Initialize XRD calculator (Standard Cu K-alpha radiation)
xrd_calc = XRDCalculator(wavelength="CuKa")
adaptor = AseAtomsAdaptor()

# Setup Plot
plt.figure(figsize=(9, 5.5))

# Loop through each dataset to process and plot
for data in datasets:
    file_path = data["file"]
    
    # 1. Load the structure from the XSF file using ASE
    try:
        atoms = read(file_path)
    except FileNotFoundError:
        print(f"Warning: Could not find '{file_path}'. Skipping this structure.")
        continue

    # Create a 3x3x3 supercell
    structure_supercell = atoms * (3, 3, 3)

    # 2. Convert to a Pymatgen structure
    pmg_structure = adaptor.get_structure(structure_supercell)

    # 3. Get XRD pattern details
    pattern = xrd_calc.get_pattern(pmg_structure, two_theta_range=(30, 50))

    # 4. Generate continuous 2-theta grid and apply peak broadening
    simulated_intensity = np.zeros_like(two_theta_grid)
    for peak_angle, intensity in zip(pattern.x, pattern.y):
        simulated_intensity += intensity * norm.pdf(two_theta_grid, peak_angle, sigma)

    # Normalize intensity to maximum 100
    if max(simulated_intensity) > 0:
        simulated_intensity = (simulated_intensity / max(simulated_intensity)) * 100

    # Add a slight background baseline so it doesn't drop to absolute 0
    simulated_intensity += 5 

    # 5. Plot the current curve using thin black lines with specific style
    plt.plot(
        two_theta_grid, 
        simulated_intensity, 
        label=data["label"], 
        color="black", 
        linestyle=data["linestyle"], 
        lw=1.2
    )

# ---------------------------------------------------
# Finalize Plot Formatting
plt.xlabel(r"$2\theta$ (degree)", fontsize=11)
plt.ylabel("Intensity (arbitrary units)", fontsize=11)
plt.title("XRD Comparison: Ta Bulk vs Boron-doped Systems", fontsize=13, pad=15)
plt.legend(frameon=True, fontsize=10)
plt.xlim(30, 50)

# Save and show
output_img = "ta_comparison_xrd.png"
plt.tight_layout()
plt.savefig(output_img, dpi=300)
plt.show()