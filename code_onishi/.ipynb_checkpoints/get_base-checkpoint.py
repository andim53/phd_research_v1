import os
import json
import numpy as np
from collections import Counter

from ase import Atoms
from ase.io import write

from agox import AGOX
from agox.environments import Environment
from agox.databases import Database
from agox.samplers import FixedSampler
from agox.collectors import ParallelCollector
from agox.acquisitors import LowerConfidenceBoundAcquisitor
from agox.postprocessors import ParallelRelaxPostprocess
from agox.evaluators import LocalOptimizationEvaluator
from agox.models.descriptors.soap import SOAP

from agox.models.GPR.kernels import RBF, Noise, Constant as C
from agox.models.GPR.priors import Repulsive
from agox.models.GPR import SparseGPREnsemble
from agox.helpers import SubprocessGPAW

from fcc_HEA_generator import HighEntropyGenerator
from HEA_permutation_generator import PermutationGenerator


# =====================
# Input JSON
# =====================

json_path = "output.json"

with open(json_path) as f:
    json_data = json.load(f)

new_json = {"structures": []}


# =====================
# Loop over structures
# =====================

for hea_idx, structure in enumerate(json_data["structures"]):

    if hea_idx > 2:
        break

    print("=" * 80)
    print(f"HEA INDEX = {hea_idx}")
    print("=" * 80)

    elements = structure["elements"]
    name = structure.get("name", "structure")

    counts = Counter(elements)
    formula = "".join(f"{el}{counts[el]}" for el in sorted(counts.keys()))

    print(f"checking for {formula}")

    # =====================
    # Parameters
    # =====================

    seed_HEA = np.random.SeedSequence().entropy
    seed_agox = 0

    num_candidates = {0: [5, 5], 26: [0, 10]}
    N_iterations = 100
    ncores = 24
    kpts = (2, 2, 2)

    base_start = 3.71
    base_step = 0.01
    n_base_loop = 1000
    n_structure_check = 50

    valid_bases = []
    best_base = None
    best_success = -1
    best_results = []

    # =====================
    # Search base loop
    # =====================

    for base_idx in range(n_base_loop):

        base = base_start + base_idx * base_step
        a = base

        print("=" * 80)
        print(f"base = {base:.4f}")
        print("=" * 80)

        # =====================
        # Output folders
        # =====================

        path_result = f"0_result/base_{base:.4f}"
        path_xsf = f"{path_result}/0_xsf"
        path_fig = f"{path_result}/1_fig"
        db_dir = f"{path_result}/db"

        for d in [path_xsf, path_fig, db_dir]:
            os.makedirs(d, exist_ok=True)

        # =====================
        # Structure generation setup
        # =====================

        cell = [
            [2 * a, 0, 0],
            [0, 2 * a, 0],
            [0, 0, 2 * a]
        ]

        template = Atoms("", cell=cell, pbc=True)

        confinement_cell = template.cell.copy()
        confinement_corner = np.array([0, 0, 0])

        frac_positions = [
            (0, 0, 0), (1/2, 0, 0), (0, 1/2, 0), (1/4, 1/4, 0),
            (3/4, 1/4, 0), (1/4, 3/4, 0), (3/4, 3/4, 0), (1/2, 1/2, 0),

            (1/4, 0, 1/4), (3/4, 0, 1/4), (0, 1/4, 1/4), (1/2, 1/4, 1/4),
            (1/4, 1/2, 1/4), (3/4, 1/2, 1/4), (0, 3/4, 1/4), (1/2, 3/4, 1/4),

            (0, 0, 1/2), (1/2, 0, 1/2), (0, 1/2, 1/2), (1/2, 1/2, 1/2),
            (1/4, 1/4, 1/2), (3/4, 1/4, 1/2), (1/4, 3/4, 1/2), (3/4, 3/4, 1/2),

            (1/4, 0, 3/4), (3/4, 0, 3/4), (0, 1/4, 3/4), (1/2, 1/4, 3/4),
            (1/4, 1/2, 3/4), (3/4, 1/2, 3/4), (0, 3/4, 3/4), (1/2, 3/4, 3/4),
        ]

        environment = Environment(
            template=template,
            symbols=formula,
            confinement_cell=confinement_cell,
            confinement_corner=confinement_corner,
            box_constraint_pbc=[True, True, True],
        )

        database = Database(
            filename=f"{db_dir}/db_{seed_agox}.db",
            order=5
        )

        # =====================
        # Generators
        # =====================

        generators = [
            HighEntropyGenerator(
                **environment.get_confinement(),
                seed=seed_HEA,
                a=a,
                cell=cell,
                frac_positions=frac_positions,
                use_initialMagmom=True,
                elements=elements,
                json_path=None,
            ),

            PermutationGenerator(
                **environment.get_confinement(),
                max_number_of_swaps=5,
                rattle_strength=0.0
            ),
        ]

        success_count = 0
        fail_count = 0

        # =====================
        # Structure check loop
        # =====================

        for i in range(n_structure_check):

            try:
                hea_candidate = generators[0](
                    sampler=None,
                    environment=environment
                )[0]

                sampler_test = FixedSampler(hea_candidate)

                permut_candidate = generators[1](
                    sampler_test,
                    environment
                )[0]

                success_count += 1

            except Exception:
                fail_count += 1

        # =====================
        # Store result
        # =====================

        best_results.append({
            "base": base,
            "success": success_count,
            "fail": fail_count,
        })

        print(f"base={base:.4f} success={success_count} fail={fail_count}")

        if fail_count == 0:
            valid_bases.append({
                "base": base,
                "success": success_count
            })
            print(f"[VALID BASE FOUND] {base:.4f} for {formula}")
            break

    # =====================
    # Save JSON result
    # =====================

    new_json["structures"].append({
        "name": name,
        "formula": formula,
        "elements": elements,
        "best_base": best_base,
        "best_success": best_success,
        "valid_bases": valid_bases
    })


# =====================
# Write output JSON
# =====================

output_path = "output_with_base.json"

with open(output_path, "w") as f:
    json.dump(new_json, f, indent=4)

print("=" * 80)
print("DONE")
print(f"Saved: {output_path}")