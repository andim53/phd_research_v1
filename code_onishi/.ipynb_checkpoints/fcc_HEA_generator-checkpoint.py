import numpy as np
from ase import Atoms
from ase.io import write
from ase.data import covalent_radii

from pathlib import Path
from datetime import datetime

import json
import random

from agox.generators.ABC_generator import GeneratorBaseClass


class HighEntropyGenerator(GeneratorBaseClass):
    """
    Creates structures without a seed and with minimal bias.

    Parameters
    -----------
    contiguous : bool
        If False, the generator may place atoms at several places on a template. If True,
        any placed atom must be placed close to an already placed atom.
        Replaces 'may_nucleate_at_several_places' which is deprecated.
    replace : bool
        ?
    """

    name = "HighEntropyGenerator"

    def __init__(
        self,
        contiguous=False,
        attempts=100,
        may_nucleate_at_several_places=None,

        seed=0,
        a=3.161058820599375,
        cell=None,
        frac_positions=None,

        elements=(
            ["Fe"] * 6 +
            ["Co"] * 6 +
            ["Ni"] * 6 +
            ["Cu"] * 4 +
            ["Pt"] * 10
        ),

        json_path=None,

        replace=True,
        use_initialMagmom=False,

        write_struct=True,
        output_dir="generated_structures",

        min_distance_scale=0.8,
        
        print_result = False,
        **kwargs
    ):

        super().__init__(replace=replace, **kwargs)

        self.contiguous = contiguous
        self.attempts = attempts

        self.seed = seed
        self.a = a
        self.cell = cell
        self.frac_positions = frac_positions
        self.elements = elements

        self.use_initialMagmom = use_initialMagmom

        self.write_struct = write_struct
        self.output_dir = Path(output_dir)
        self.run_dir = self._create_new_run_dir() if write_struct else None

        self.min_distance_scale = min_distance_scale

        self.json_data = None

        self.print_result = print_result


        if json_path is not None:
            with open(json_path) as f:
                self.json_data = json.load(f)

        if may_nucleate_at_several_places is not None:
            DeprecationWarning(
                "'may_nucleate_at_several_places' is deprecated and will be removed. "
                "Use 'contiguous' instead."
            )

            self.contiguous = not may_nucleate_at_several_places

    def check_too_close(self, atoms, scale=None):
        """
        Check whether any atom pair is too close.

        Uses covalent radii and periodic boundary conditions.

        Parameters
        ----------
        atoms : ASE Atoms
            Structure to check.

        scale : float
            Multiplication factor for covalent radii sum.
            If None, uses self.min_distance_scale.

        Returns
        -------
        bool
            True if atoms are too close.
        """

        if scale is None:
            scale = self.min_distance_scale

        numbers = atoms.get_atomic_numbers()

        # Distance matrix using minimum image convention
        dmat = atoms.get_all_distances(mic=True)

        # Ignore self-distances
        np.fill_diagonal(dmat, np.inf)

        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):

                cutoff = scale * (
                    covalent_radii[numbers[i]] +
                    covalent_radii[numbers[j]]
                )

                if dmat[i, j] < cutoff:
                    return True

        return False

    def _get_candidates(self, candidate, parents, environment):

        rng = np.random.default_rng(self.seed)

        a = self.a
        cell = self.cell
        frac_positions = self.frac_positions

        # -------------------------------------------------------------
        # Choose element set
        # -------------------------------------------------------------

        if self.json_data is not None:

            structure = random.choice(self.json_data["structures"])

            elements = structure["elements"]
            name = structure.get("name", "structure")

        else:
            elements = list(self.elements).copy()
            name = "random_structure"

        # -------------------------------------------------------------
        # Basic checks
        # -------------------------------------------------------------

        assert frac_positions is not None
        assert cell is not None

        assert len(elements) == len(frac_positions), (
            f"Number of elements ({len(elements)}) does not match "
            f"number of positions ({len(frac_positions)})"
        )

        # -------------------------------------------------------------
        # Shuffle element ordering
        # -------------------------------------------------------------

        rng.shuffle(elements)

        # -------------------------------------------------------------
        # Build structure
        # -------------------------------------------------------------

        atoms = Atoms(
            symbols=elements,
            scaled_positions=frac_positions,
            cell=cell,
            pbc=True
        )

        # -------------------------------------------------------------
        # Distance validation
        # -------------------------------------------------------------

        if self.check_too_close(atoms):
            if self.print_result:
                self.writer(
                    "Rejected structure because some atoms are too close"
                )

            return []

        # -------------------------------------------------------------
        # Add to candidate
        # -------------------------------------------------------------

        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()

        candidate.extend(
            Atoms(
                numbers=atomic_numbers,
                positions=positions,
                cell=cell,
                pbc=True
            )
        )

        # -------------------------------------------------------------
        # Initial magnetic moments
        # -------------------------------------------------------------

        if self.use_initialMagmom:

            candidate.set_initial_magnetic_moments(
                [
                    2.0 if el == "Mn" else 0.0
                    for el in atoms.get_chemical_symbols()
                ]
            )

        # -------------------------------------------------------------
        # Write structure
        # -------------------------------------------------------------

        if self.write_struct:

            self.structure_counter += 1

            filename = (
                self.run_dir /
                f"hea_rand_{self.structure_counter:04d}.xsf"
            )

            write(filename, candidate)

        return [candidate]

    def _create_new_run_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.structure_counter = 0
        print(f"Saved to: {run_dir}")
        return run_dir

    def get_number_of_parents(self, sampler):
        return 0