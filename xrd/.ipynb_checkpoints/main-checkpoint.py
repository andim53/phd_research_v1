import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from scipy.stats import norm

# ==========================================
# CONFIGURATION OPTION
# Choose between "W" (Tungsten) or "Ta" (Tantalum)
# ==========================================
material_option = "W"  # Set to "W" or "Ta"
# ==========================================

# 1. Create the selected bcc crystal structure in ASE
if material_option == "W":
    # Experimental lattice constant ~3.16 Å
    atoms = bulk("W", "bcc", a=3.16, cubic=True)
    label_name = r"$\alpha$-W"
    title_name = "Tungsten (W)"
elif material_option == "Ta":
    # Experimental lattice constant ~3.30 Å
    atoms = bulk("Ta", "bcc", a=3.30, cubic=True)
    label_name = r"$\alpha$-Ta"
    title_name = "Tantalum (Ta)"
else:
    raise ValueError("Invalid material_option. Choose 'W' or 'Ta'.")

# Create a 3x3x3 supercell (matching your original logic)
structure_supercell = atoms * (3, 3, 3)

# 2. Convert to a Pymatgen structure
adaptor = AseAtomsAdaptor()
pmg_structure = adaptor.get_structure(structure_supercell)

# 3. Initialize XRD calculator (Standard Cu K-alpha radiation)
xrd_calc = XRDCalculator(wavelength="CuKa")
pattern = xrd_calc.get_pattern(pmg_structure, two_theta_range=(30, 50))

# 4. Generate continuous 2-theta grid and apply peak broadening (FWHM)
two_theta_grid = np.linspace(30, 50, 1000)
simulated_intensity = np.zeros_like(two_theta_grid)

# Setting peak width
fwhm = 0.1  
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

for peak_angle, intensity in zip(pattern.x, pattern.y):
    simulated_intensity += intensity * norm.pdf(two_theta_grid, peak_angle, sigma)

# Normalize intensity
if max(simulated_intensity) > 0:
    simulated_intensity = (simulated_intensity / max(simulated_intensity)) * 100

# Add a slight background baseline so it doesn't drop to absolute 0
simulated_intensity += 5 

# ---------------------------------------------------
# Plotting the replication
plt.figure(figsize=(8, 5))
plt.plot(two_theta_grid, simulated_intensity, label=f"Simulated {label_name} (x=0)", color="black", lw=2)
plt.xlabel(r"$2\theta$ (degree)")
plt.ylabel("Intensity (arbitrary units)")
plt.title(f"Replication of Crystalline {title_name} Peak")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
output_img = "ta_comparison_xrd.png"
plt.savefig(output_img, dpi=300)
plt.show()