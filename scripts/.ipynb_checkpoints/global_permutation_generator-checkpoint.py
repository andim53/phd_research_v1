
from typing import List
import numpy as np
from ase.io import write

from agox.candidates import Candidate
from agox.environments import Environment
from agox.generators.ABC_generator import GeneratorBaseClass
from agox.samplers import Sampler


class GlobalPermutationGenerator(GeneratorBaseClass):
    """
    Permutes atoms of different species in a seed structure and rattles them 
    using the native AGOX check system.

    Parameters
    ----------
    max_number_of_swaps : int
        Maximum number of permutations to perform.
    rattle_strength : float
        Maximum distance to rattle atoms.
    ignore_H : bool
        If True, ignore hydrogen atoms.
    write_candidates_to_disk : bool
        If True, write candidates to disk.
    attempts : int
        Number of geometric configuration attempts to perform per swap.
    """

    name = "PermutationGenerator"

    def __init__(
        self,
        max_number_of_swaps: int = 1,
        rattle_strength: float = 0.0,
        ignore_H: bool = False,
        write_candidates_to_disk: bool = False,
        replace: bool = True,
        attempts: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(replace=replace, **kwargs)
        self.max_number_of_swaps = max_number_of_swaps
        self.rattle_strength = rattle_strength
        self.ignore_H = ignore_H
        self.write_candidates_to_disk = write_candidates_to_disk
        self.attempts = attempts

    def _get_candidates(self, candidate: Candidate, parents: List[Candidate], environment: Environment):
        if self.write_candidates_to_disk:
            write(f"candidate_{self.counter}.traj", candidate)

        num_template_atoms = len(candidate.get_template())

        # Extract symbols exclusively for non-template active atoms
        active_numbers = candidate.get_atomic_numbers()[num_template_atoms:]
        unique_numbers = np.unique(active_numbers)

        if self.ignore_H:
            unique_numbers = unique_numbers[unique_numbers != 1]

        assert len(unique_numbers) > 1, "Cannot be used for single component systems"

        num_swaps = np.random.randint(self.max_number_of_swaps) + 1

        for n in range(num_swaps):
            # Select two distinct random element species to swap
            num_species = len(unique_numbers)
            idx_i = np.random.randint(num_species)
            num_i = unique_numbers[idx_i]

            remaining_numbers = np.delete(unique_numbers, idx_i)
            num_j = np.random.choice(remaining_numbers)

            # Get complete absolute index arrays matching the selected species
            indices_i = np.where(active_numbers == num_i)[0] + num_template_atoms
            indices_j = np.where(active_numbers == num_j)[0] + num_template_atoms

            # Create an executable permutation grid of candidate pairs
            combinations = np.array(np.meshgrid(indices_i, indices_j)).T.reshape(-1, 2)
            shuffled_combinations = np.random.permutation(combinations)

            swap_occurred = False
            for swap_idx_i, swap_idx_j in shuffled_combinations:
                
                # Base swapped positions
                base_positions = candidate.get_positions().copy()
                base_positions[[swap_idx_i, swap_idx_j]] = base_positions[[swap_idx_j, swap_idx_i]]

                valid_rattle = False
                
                # Dynamic positional adjustment and checking sequence
                for _ in range(self.attempts):
                    # Generate candidate displacement offsets for both targeted nodes
                    radius_i = self.rattle_strength * np.random.rand() ** (1 / self.get_dimensionality())
                    radius_j = self.rattle_strength * np.random.rand() ** (1 / self.get_dimensionality())
                    
                    suggested_pos_i = base_positions[swap_idx_i] + self.get_displacement_vector(radius_i)
                    suggested_pos_j = base_positions[swap_idx_j] + self.get_displacement_vector(radius_j)

                    # Check confinement domain limits
                    if not self.check_confinement(suggested_pos_i).all() or not self.check_confinement(suggested_pos_j).all():
                        continue

                    # Temporarily apply positions to perform structural criteria tests
                    original_positions = candidate.get_positions().copy()
                    
                    candidate[swap_idx_i].position = suggested_pos_i
                    candidate[swap_idx_j].position = suggested_pos_j

                    # Validate hard-sphere steric bounds with other atoms
                    check_i = self.check_new_position(
                        candidate, suggested_pos_i, candidate[swap_idx_i].number, skipped_indices=[swap_idx_i]
                    )
                    check_j = self.check_new_position(
                        candidate, suggested_pos_j, candidate[swap_idx_j].number, skipped_indices=[swap_idx_j]
                    )

                    if check_i and check_j:
                        valid_rattle = True
                        break
                    else:
                        # Rollback if structural checks fail
                        candidate.set_positions(original_positions)

                if valid_rattle:
                    if self.write_candidates_to_disk:
                        write(f"candidate_swap_{n}_{self.counter}.traj", candidate)
                    swap_occurred = True
                    break

            if not swap_occurred:
                self.writer("No swaps possible")
                return []

        return [candidate]

    def get_number_of_parents(self, sampler: Sampler) -> int:
        return 1