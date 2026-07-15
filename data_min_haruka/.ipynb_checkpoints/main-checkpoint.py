"""
FLAPW Workflow Script
=====================

Version
-------
v0.0.7
- Pure ASE workflow: Read structures from a .traj file.
- Directory names created using index and ASE chemical formulas only.
- Removed pymatgen dependencies.
"""

import shutil
import traceback
from pathlib import Path

from ase.io import read, write
from flapw import FLAPW


# =============================================================================
# Configuration
# =============================================================================

BIN_FLAPW = "flapw"

# False : serial calculation (flapw)
# True  : MPI calculation (pflapw)
MPI = False

# First structure index
START = 15

# Number of structures processed in one execution
NSTRUCTURES = 3

# Trajectory file name to read structures from
TRAJ_FILE = "structures.traj"


# =============================================================================
# Load structure database (from traj file)
# =============================================================================

# Read all atoms from the trajectory file
atoms_list = read(TRAJ_FILE, index=":")

# Process only the specified range
END = min(START + NSTRUCTURES, len(atoms_list))

results = []


# =============================================================================
# Main Loop
# =============================================================================

for i, atoms in enumerate(atoms_list[START:END], start=START):

    formula = "Unknown"
    dirname = f"job_{i}"

    try:

        print(f"\n{'='*60}")
        print(f"Processing Index : {i}")
        print(f"{'='*60}")

        # =====================================================================
        # Phase 1 : Structure preparation
        # =====================================================================

        # Get empirical chemical formula directly from ASE
        formula = atoms.get_chemical_formula(mode="hill")
        dirname = f"{i}_{formula}"

        workdir = Path(dirname)
        workdir.mkdir(exist_ok=True)

        # Copy executable
        shutil.copy2(BIN_FLAPW, workdir / BIN_FLAPW)

        # Save structure for visualization/debugging
        write(workdir / "atoms.xsf", atoms)

        # =====================================================================
        # Phase 2 : Ground-state SCF calculation
        # =====================================================================

        atoms.calc = FLAPW(
            mpi=MPI,
            directory=str(workdir),
        )

        print("[Phase 2] Ground-state SCF")

        energy = atoms.get_potential_energy()

        print(f"{dirname} : {energy:.6f} eV")

        results.append(
            {
                "formula": formula,
                "energy": energy,
                "directory": dirname,
            }
        )

        # =====================================================================
        # Phase 3 : SOC calculation
        # =====================================================================

        print("[Phase 3] SOC calculation")

        # Copy SCF results
        atoms.calc.copy_scf()
        print("SCF copied.")

        # Modify lapwin for SOC calculation
        atoms.calc.prepare_soc_lapwin()

        print("Updated lapwin")
        print("-" * 60)
        print((workdir / "lapwin").read_text())
        print("-" * 60)

        # ---------------------------------------------------------------------
        # Explicitly execute the restart calculation
        # ---------------------------------------------------------------------
        atoms.calc.execute()

        # Save SOC results
        atoms.calc.copy_soc()
        print("SOC copied.")

        # =====================================================================
        # Phase 4 : Optics calculation
        # =====================================================================

        print("[Phase 4] Optics")

        # Prepare optics input files
        atoms.calc.prepare_optics()
        print("Optics input prepared.")

        # Execute optics calculation
        atoms.calc.run_xoptics()
        print("xoptics finished.")

        # Optional cleanup
        atoms.calc.clean()

    except Exception as e:

        print(f"FAILED : {formula}")
        print(e)
        traceback.print_exc()

        results.append(
            {
                "formula": formula,
                "energy": None,
                "directory": dirname,
                "error": str(e),
            }
        )

        continue


# =============================================================================
# Summary
# =============================================================================

print("\nFinished All Processes.")

print(f"Structures processed : {END - START}")
print(f"Successful           : {sum(r['energy'] is not None for r in results)}")
print(f"Failed               : {sum(r['energy'] is None for r in results)}")