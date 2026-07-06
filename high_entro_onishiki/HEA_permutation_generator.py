import numpy as np

from ase.data import covalent_radii
from ase.io import write

from pathlib import Path
from datetime import datetime

from agox.generators.ABC_generator import GeneratorBaseClass


class PermutationGenerator(GeneratorBaseClass):
    """
    Permutes atoms of different species in a seed structure.

    Parameters
    ----------
    max_number_of_swaps : int
        Maximum number of swaps.

    rattle_strength : float
        Random displacement amplitude.

    use_xy_only : bool
        If True, only rattle in xy-plane.

    ignore_H : bool
        Ignore hydrogen atoms.

    write_candidates_to_disk : bool
        Save intermediate traj files.

    min_distance_scale : float
        Minimum allowed distance factor.

    max_distance_scale : float
        Defines whether swapped atoms remain close enough
        to neighboring atoms.
    """

    name = "PermutationGenerator"

    def __init__(
        self,
        max_number_of_swaps=1,
        rattle_strength=0.0,
        use_xy_only=False,
        ignore_H=False,
        write_candidates_to_disk=False,
        replace=True,
        write_struct=True,
        output_dir="generated_structures",

        min_distance_scale=0.85,
        max_distance_scale=1.15,
        
        print_result = False,
        

        **kwargs,
    ):

        super().__init__(replace=replace, **kwargs)

        self.max_number_of_swaps = max_number_of_swaps
        self.rattle_strength = rattle_strength

        self.ignore_H = ignore_H
        self.use_xy_only = use_xy_only

        self.write_candidates_to_disk = write_candidates_to_disk

        self.write_struct = write_struct
        self.output_dir = Path(output_dir)
        self.run_dir = self._create_new_run_dir() if write_struct else None

        self.min_distance_scale = min_distance_scale
        self.max_distance_scale = max_distance_scale
        
        self.print_result = print_result

    # -----------------------------------------------------------------
    # Swap validation
    # -----------------------------------------------------------------

    def validate_swap(
        self,
        candidate,
        new_positions,
        swap_idx_i,
        swap_idx_j,
    ):
        """
        Validate whether swapped positions are physically reasonable.

        Conditions:
        -----------
        1. Atoms must not overlap.
        2. At least one neighboring atom must remain near enough.

        Returns
        -------
        bool
            True if swap is acceptable.
        """

        swap_successfull = True
        near_enough_to_other_atoms = False

        for i in (swap_idx_i, swap_idx_j):

            for other_atom_idx in range(len(candidate)):

                if other_atom_idx in [swap_idx_i, swap_idx_j]:
                    continue

                other_atom = candidate[other_atom_idx]

                covalent_dist = (
                    covalent_radii[candidate[i].number] +
                    covalent_radii[other_atom.number]
                )

                rmin = self.min_distance_scale * covalent_dist
                rmax = self.max_distance_scale * covalent_dist

                dist = np.linalg.norm(
                    other_atom.position - new_positions[i]
                )

                # -----------------------------------------------------
                # Too close -> reject
                # -----------------------------------------------------

                if dist < rmin:
                    swap_successfull = False
                    break

                # -----------------------------------------------------
                # At least one nearby atom required
                # -----------------------------------------------------

                if dist < rmax:
                    near_enough_to_other_atoms = True

            if not swap_successfull or not near_enough_to_other_atoms:
                break

        if not swap_successfull:
            return False

        if not near_enough_to_other_atoms:
            return False

        return True

    # -----------------------------------------------------------------
    # Candidate generation
    # -----------------------------------------------------------------

    def _get_candidates(self, candidate, parents, environment):

        if self.write_candidates_to_disk:
            write(f"candidate_{self.counter}.traj", candidate)

        template = candidate.get_template()

        number_of_atoms = len(candidate)
        number_of_template_atoms = len(template)
        number_of_non_template_atoms = (
            number_of_atoms - number_of_template_atoms
        )

        symbols = candidate.get_atomic_numbers()[number_of_template_atoms:]

        unique_symbols = np.unique(symbols)

        if self.ignore_H:
            unique_symbols = np.array(
                [s for s in unique_symbols if s != 1]
            )

        number_of_unique_symbols = len(unique_symbols)

        assert (
            number_of_unique_symbols > 1
        ), "Cannot be used for single component systems"

        number_of_swaps = (
            np.random.randint(self.max_number_of_swaps) + 1
        )

        # -----------------------------------------------------------------
        # Perform swaps
        # -----------------------------------------------------------------

        for n in range(number_of_swaps):

            symbol_i = unique_symbols[
                np.random.randint(number_of_unique_symbols)
            ]

            remaining_symbols = np.delete(
                unique_symbols,
                [
                    idx
                    for idx in range(number_of_unique_symbols)
                    if unique_symbols[idx] == symbol_i
                ]
            )

            symbol_j = remaining_symbols[
                np.random.randint(number_of_unique_symbols - 1)
            ]

            idx_symbol_i = (
                np.argwhere(symbols == symbol_i).reshape(-1)
                + number_of_template_atoms
            )

            idx_symbol_j = (
                np.argwhere(symbols == symbol_j).reshape(-1)
                + number_of_template_atoms
            )

            combinations_ij = np.array(
                np.meshgrid(idx_symbol_i, idx_symbol_j)
            ).T.reshape(-1, 2)

            # -------------------------------------------------------------
            # Try swap combinations
            # -------------------------------------------------------------

            for row in np.random.permutation(combinations_ij):

                swap_idx_i = row[0]
                swap_idx_j = row[1]

                # ---------------------------------------------------------
                # Create swapped positions
                # ---------------------------------------------------------

                new_positions = candidate.get_positions().copy()

                new_positions[[swap_idx_i, swap_idx_j]] = (
                    new_positions[[swap_idx_j, swap_idx_i]]
                )

                # ---------------------------------------------------------
                # Apply rattling
                # ---------------------------------------------------------

                if self.use_xy_only:

                    new_positions[swap_idx_i] += (
                        self.pos_add_disk(self.rattle_strength)
                    )

                    new_positions[swap_idx_j] += (
                        self.pos_add_disk(self.rattle_strength)
                    )

                else:

                    new_positions[swap_idx_i] += (
                        self.pos_add_sphere(self.rattle_strength)
                    )

                    new_positions[swap_idx_j] += (
                        self.pos_add_sphere(self.rattle_strength)
                    )

                # ---------------------------------------------------------
                # Validate swap
                # ---------------------------------------------------------

                swap_valid = self.validate_swap(
                    candidate,
                    new_positions,
                    swap_idx_i,
                    swap_idx_j,
                )

                if not swap_valid:
                    # print(f"{swap_valid=}")
                    continue

                # ---------------------------------------------------------
                # Accept swap
                # ---------------------------------------------------------

                candidate.set_positions(new_positions)

                if self.write_candidates_to_disk:
                    write(
                        f"candidate_swap_{n}_{self.counter}.traj",
                        candidate,
                    )

                break

            else:
                if self.print_result:
                    self.writer("No swaps possible")
                return []

        # -----------------------------------------------------------------
        # Save final structure
        # -----------------------------------------------------------------

        if self.write_struct:

            self.structure_counter += 1

            filename = (
                self.run_dir /
                f"hea_permut_{self.structure_counter:04d}.xsf"
            )

            write(filename, candidate)

        return [candidate]

    # -----------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------

    def _create_new_run_dir(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_dir = self.output_dir / f"run_{timestamp}"

        run_dir.mkdir(parents=True, exist_ok=True)

        self.structure_counter = 0

        print(f"Saved to: {run_dir}")

        return run_dir

    # -----------------------------------------------------------------
    # Rattle helper functions
    # -----------------------------------------------------------------

    def pos_add_disk(self, rattle_strength):
        """Random displacement within disk"""

        r = rattle_strength * np.random.rand() ** (1 / 2)

        theta = np.random.uniform(low=0, high=2 * np.pi)

        pos_add = r * np.array([
            np.cos(theta),
            np.sin(theta),
            0,
        ])

        return pos_add

    def pos_add_sphere(self, rattle_strength):
        """Random displacement within sphere"""

        r = rattle_strength * np.random.rand() ** (1 / 3)

        theta = np.random.uniform(low=0, high=2 * np.pi)
        phi = np.random.uniform(low=0, high=np.pi)

        pos_add = r * np.array([
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ])

        return pos_add

    # -----------------------------------------------------------------
    # AGOX requirement
    # -----------------------------------------------------------------

    def get_number_of_parents(self, sampler):
        return 1