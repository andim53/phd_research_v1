"""
FLAPW Workflow Script
=====================

Version
-------
v0.0.5
- Process only a specified number of structures (`NSTRUCTURES`)
  in a single execution.
- Useful for dividing long workflows into multiple sequential runs.

v0.0.4
- Added `FLAPW.execute()` to explicitly execute restart calculations
  (SOC calculation no longer relies on `get_potential_energy()`).

Workflow
--------
1. Ground-state SCF calculation
2. Copy SCF results
3. SOC restart calculation (`execute()`)
4. Copy SOC results
5. Optics calculation

This script performs:
    1. Load crystal structures from a pickle file.
    2. Convert structures to primitive cells.
    3. Run ground-state SCF calculations.
    4. Continue from SCF and perform SOC calculations.
    5. Perform optics calculations.
"""

import pickle
import shutil
import traceback
from pathlib import Path

from ase.io import write
from flapw import FLAPW
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


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


# =============================================================================
# Load structure database
# =============================================================================

with open("festable_docs.pkl", "rb") as f:
    docs_dict = pickle.load(f)

docs = list(docs_dict)

# Process only the specified range
END = min(START + NSTRUCTURES, len(docs))

results = []


# =============================================================================
# Main Loop
# =============================================================================

for i, doc in enumerate(docs[START:END], start=START):

    formula = "Unknown"
    dirname = f"job_{i}"

    try:

        print(f"\n{'='*60}")
        print(f"Processing Index : {i}")
        print(f"{'='*60}")

        # =====================================================================
        # Phase 1 : Structure preparation
        # =====================================================================

        structure = Structure.from_dict(doc["structure"])
        formula = structure.composition.reduced_formula

        if "material_id" in doc:
            dirname = f"{i}_{formula}_{doc['material_id']}"
        else:
            dirname = f"{i}_{formula}"

        workdir = Path(dirname)
        workdir.mkdir(exist_ok=True)

        # Copy executable
        shutil.copy2(BIN_FLAPW, workdir / BIN_FLAPW)

        # Convert to primitive cell to reduce computational cost
        sga = SpacegroupAnalyzer(structure, symprec=1e-3)
        primitive = sga.find_primitive()

        if primitive is not None:
            structure = primitive

        atoms = AseAtomsAdaptor.get_atoms(structure)

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
        # v0.0.4
        #
        # Explicitly execute the restart calculation.
        # Previous versions restarted using get_potential_energy().
        # The execute() method performs only the calculation execution.
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
        #
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
