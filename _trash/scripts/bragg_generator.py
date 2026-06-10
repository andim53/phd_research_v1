import numpy as np
import random
from pathlib import Path
from datetime import datetime

from ase import Atoms
from ase.io import write
from ase.data import covalent_radii

from agox.generators.ABC_generator import GeneratorBaseClass


class BraggGenerator(GeneratorBaseClass):
    """
    Generates structures with controlled long-range order (S) directly from an 
    atoms template (e.g., a pre-built Fe3Pt supercell).

    Parameters
    ----------
    atoms_template : ASE Atoms
        The pristine, ordered structural template to introduce anti-site disorder into.
    S_target : float
        Target long-range order parameter (between -1.0 and 1.0).
        If None, a random choice from S_options is picked per generation step.
    S_options : list of float
        A pool of S values to randomly sample from if S_target is None.
    seed : int
        Initial random seed for the generation process if randomize_seed is False.
    randomize_seed : bool
        If True, ignores the seed parameter and generates a completely random initial seed
        using system entropy.
    write_struct : bool
        If True, writes the generated structures to disk.
    output_dir : str or Path
        Directory where structures are saved.
    min_distance_scale : float
        Multiplication factor for covalent radii sum to check for unphysical collisions.
    print_result : bool
        If True, prints details of the generated candidate.
    """

    name = "BraggGenerator"

    def __init__(
        self,
        atoms_template,
        S_target=1.0,
        S_options=None,
        seed=2,
        randomize_seed=False,
        write_struct=True,
        output_dir="generated_structures",
        min_distance_scale=0.8,
        print_result=False,
        replace=True,
        **kwargs,
    ):
        super().__init__(replace=replace, **kwargs)

        self.atoms_template = atoms_template.copy()
        self.S_target = S_target
        self.S_options = S_options if S_options is not None else [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        
        self.randomize_seed = randomize_seed
        if self.randomize_seed:
            # Grab high-quality systemic entropy for initialization
            self.seed = int(np.random.SeedSequence().entropy & 0xFFFFFFFF)
        else:
            self.seed = seed

        self.write_struct = write_struct
        self.output_dir = Path(output_dir)
        self.run_dir = self._create_new_run_dir() if write_struct else None

        self.min_distance_scale = min_distance_scale
        self.print_result = print_result

        # -------------------------------------------------------------
        # Map Ideal Site Layout Directly From Template
        # -------------------------------------------------------------
        self.ideal_species = {i: sym for i, sym in enumerate(self.atoms_template.get_chemical_symbols())}

        # Cache reference positions and cell geometry to reuse
        self.cell = self.atoms_template.get_cell()
        self.positions = self.atoms_template.get_positions()

        # Collect base ideal sites
        self.fe_sites = [i for i, sym in self.ideal_species.items() if sym == "Fe"]
        self.pt_sites = [i for i, sym in self.ideal_species.items() if sym == "Pt"]

    def check_too_close(self, atoms, scale=None):
        if scale is None:
            scale = self.min_distance_scale

        numbers = atoms.get_atomic_numbers()
        dmat = atoms.get_all_distances(mic=True)
        np.fill_diagonal(dmat, np.inf)

        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                cutoff = scale * (covalent_radii[numbers[i]] + covalent_radii[numbers[j]])
                if dmat[i, j] < cutoff:
                    return True
        return False

    def number_of_swaps_for_S(self, total_atoms, S):
        """Calculates anti-site swaps required to hit a specific S value."""
        r = (S + 1.0) / 2.0
        wrong_sites = int(round(total_atoms * (1.0 - r)))
        return wrong_sites // 2

    def calculate_S(self, symbols):
        """Calculates actual long-range order param based on dynamic symbols."""
        correct_sites = sum(1 for i, sym in enumerate(symbols) if sym == self.ideal_species[i])
        r = correct_sites / len(symbols)
        return 2.0 * r - 1.0

    def _get_candidates(self, candidate, parents, environment):
        # Determine target S for this call iteration
        S_target = self.S_target if self.S_target is not None else random.choice(self.S_options)

        # Initialize local pseudo-random number generator using the instance seed
        rng = np.random.default_rng(self.seed)
        # Advance seed to avoid identical sequences on back-to-back generator iterations
        self.seed = int(rng.integers(0, 2**32 - 1))

        # Reset structure mapping from pristine configuration
        symbols = list(self.atoms_template.get_chemical_symbols()).copy()
        total_atoms = len(symbols)
        swaps_needed = self.number_of_swaps_for_S(total_atoms, S_target)

        # Identify correctly occupied sites ready for anti-site operations
        fe_correct = [i for i in self.fe_sites if symbols[i] == "Fe"]
        pt_correct = [i for i in self.pt_sites if symbols[i] == "Pt"]

        # Ensure we don't request more swaps than available sites
        swaps_needed = min(swaps_needed, len(fe_correct), len(pt_correct))

        # Random sample utilizing python's random space seeded via rng choice
        local_random = random.Random(self.seed)
        chosen_fe = local_random.sample(fe_correct, swaps_needed)
        chosen_pt = local_random.sample(pt_correct, swaps_needed)

        # Swap elements to destroy order targetively
        for fe_i, pt_i in zip(chosen_fe, chosen_pt):
            symbols[fe_i] = "Pt"
            symbols[pt_i] = "Fe"

        # Build clean candidate structure state
        atoms = Atoms(symbols=symbols, positions=self.positions, cell=self.cell, pbc=True)

        # Check for unphysical collisions
        if self.check_too_close(atoms):
            if self.print_result:
                self.writer("Rejected structure because some atoms are too close")
            return []

        # Merge structural information into candidate wrapper object
        candidate.extend(atoms)

        # Compute diagnostic actual S value
        S_actual = self.calculate_S(symbols)

        if self.print_result:
            self.writer(f"Generated structured Fe3Pt: Target S={S_target}, Actual S={S_actual:.4f} (Seed={self.seed})")

        # -------------------------------------------------------------
        # Write structure out
        # -------------------------------------------------------------
        if self.write_struct:
            self.structure_counter += 1
            filename = self.run_dir / f"Fe3Pt_S_{S_actual:.2f}_{self.structure_counter:04d}.xsf"
            write(filename, candidate)

        return [candidate]

    def _create_new_run_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_ordered_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.structure_counter = 0
        print(f"Ordered structures tracking directory: {run_dir}")
        return run_dir

    def get_number_of_parents(self, sampler):
        return 0