from flapw_python import FLAPW

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pathlib import Path
from ase.io import write

import pickle
import shutil
import traceback

with open("festable_docs.pkl", "rb") as f:
    docs_dict = pickle.load(f)

results = []

for i, doc in enumerate(docs_dict):

    try:

        print(f"\n===== {i} =====")

        structure = Structure.from_dict(
            doc["structure"]
        )

        formula = (
            structure.composition.reduced_formula
        )

        # MP-ID があれば付ける
        if "material_id" in doc:
            dirname = (
                f"{i}_{formula}_{doc['material_id']}"
            )
        else:
            dirname = f"{i}_{formula}"

        workdir = Path(dirname)
        workdir.mkdir(exist_ok=True)

        #
        # flapw executable
        #
        shutil.copy2(
            "flapw",
            workdir / "flapw"
        )

        #
        # primitive cell
        #
        sga = SpacegroupAnalyzer(
            structure,
            symprec=1e-3
        )

        primitive = sga.find_primitive()

        if primitive is not None:
            structure = primitive

        atoms = AseAtomsAdaptor.get_atoms(
            structure
        )

        #
        # xsf
        #
        write(
            workdir / "atoms.xsf",
            atoms
        )

        atoms.calc = FLAPW(
            command="./flapw",
            directory=str(workdir),
        )

        energy = (
            atoms.get_potential_energy()
        )

        print(
            f"{dirname}: "
            f"{energy:.6f} eV"
        )

        results.append(
            {
                "formula": formula,
                "energy": energy,
                "directory": dirname,
            }
        )

    except Exception as e:

        print(
            f"FAILED: {formula}"
        )

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

print("\nFinished.")
print(
    f"Successful: "
    f"{sum(r['energy'] is not None for r in results)}"
)

print(
    f"Failed: "
    f"{sum(r['energy'] is None for r in results)}"
)
