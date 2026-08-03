import random
import numpy as np
import matplotlib.pyplot as plt
from ase.cluster.cubic import BodyCenteredCubic
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from scipy.stats import norm

# 1. Create the small finite bcc-W cluster in ASE
atoms = BodyCenteredCubic('W', [(1, 0, 0), (0, 1, 0), (0, 0, 1)], [3, 3, 3], latticeconstant=3.16)

# 2. Amorphization: Displace atoms randomly to break rigid periodicity
random.seed(42)
for atom in atoms:
    atom.position += np.array([random.uniform(-0.35, 0.35) for _ in range(3)])

# 3. Substitute ~32% of atoms with Ta to match chemistry
num_to_substitute = int(len(atoms) * 0.32)
indices_to_sub = random.sample(range(len(atoms)), num_to_substitute)
for idx in indices_to_sub:
    atoms[idx].symbol = 'Ta'

# --- THE FIX: Give the isolated cluster an artificial periodic bounding box ---
# This defines a 30 Å cubic cell and places the cluster right in the middle,
# giving Pymatgen a non-zero, non-singular lattice matrix to invert safely.
atoms.center(vacuum=15.0)
# ------------------------------------------------------------------------------

# 4. Convert the disordered cluster to a Pymatgen structure object
adaptor = AseAtomsAdaptor()
pmg_cluster = adaptor.get_structure(atoms)

# 5. Initialize Pymatgen's XRD Calculator
xrd_calc = XRDCalculator(wavelength="CuKa")
pattern = xrd_calc.get_pattern(pmg_cluster, two_theta_range=(30, 50))

# 6. Generate the continuous 2-theta grid
two_theta_grid = np.linspace(30, 50, 1000)
amorphous_intensities = np.zeros_like(two_theta_grid)

# --- BROADENING TO SIMULATE THE AMORPHOUS HUMP ---
# Using a very wide FWHM matches the broad hump in the paper
fwhm = 0.1 #7.0  
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

for peak_angle, intensity in zip(pattern.x, pattern.y):
    amorphous_intensities += intensity * norm.pdf(two_theta_grid, peak_angle, sigma)

# Safe normalization (prevents division-by-zero crashes)
max_val = max(amorphous_intensities)
if max_val > 0:
    amorphous_intensities = (amorphous_intensities / max_val) * 100

# Add baseline experimental floor noise
amorphous_intensities += 5

# 7. Plotting the amorphous profile
plt.figure(figsize=(6, 5))
plt.plot(two_theta_grid, amorphous_intensities, label="Simulated Amorphous W-Ta-B", color="red", lw=2)

# Styling details matching your previous graphs
plt.xlim(30, 50)
plt.ylim(0, 120)
plt.xlabel(r"$2\theta$ (degree)", fontsize=12)
plt.ylabel("Intensity (arbitrary units)", fontsize=12)
plt.title("Replication of Amorphous Broad Hump", fontsize=12, fontweight='bold')
plt.legend(frameon=False)
plt.tick_params(direction='in', top=True, right=True)
plt.tight_layout()
plt.show()