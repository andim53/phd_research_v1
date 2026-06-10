import numpy as np
from pathlib import Path
from datetime import datetime

from ase.io import write
from ase.data import covalent_radii, atomic_numbers
from agox.generators.ABC_generator import GeneratorBaseClass

class AmorphPermutationGenerator(GeneratorBaseClass):
    """
    Permutes atoms of different species in a candidate structure with rattling,
    while validating local geometry bounds.
    """

    name = "AmorphPermutationGenerator"
    __version__ = "0.0.4"

    def __init__(
        self,
        max_number_of_swaps=1,
        rattle_strength=0.0,
        use_xy_only=False,
        ignore_species=None,
        write_candidates_to_disk=False,
        replace=True,
        attempts=100,
        check_overlap=False,
        
        # Distance scale parameters used for overlap validations
        min_distance_scale=0.85,
        max_distance_scale=1.15,
        print_result=False,

        output_dir: str = "generated_structures", 
        write_struct: bool = True, # Save generated structures to disk
        **kwargs,
    ):
        super().__init__(replace=replace, **kwargs)
        self.max_number_of_swaps = max_number_of_swaps
        self.rattle_strength = rattle_strength
        self.use_xy_only = use_xy_only
        self.write_candidates_to_disk = write_candidates_to_disk
        self.attempts = attempts
        self.check_overlap = check_overlap
        
        self.min_distance_scale = min_distance_scale
        self.max_distance_scale = max_distance_scale
        self.print_result = print_result

        self.output_dir = Path(output_dir)
        self.structure_counter = 0
        self.run_dir = self._create_new_run_dir() if write_struct else None

        self.write_struct = write_struct
        
        self.ignored_numbers = []
        if ignore_species is not None:
            for s in ignore_species:
                if isinstance(s, str):
                    self.ignored_numbers.append(atomic_numbers[s])
                else:
                    self.ignored_numbers.append(s)
        
    def validate_swap(self, candidate, new_positions, swap_idx_i, swap_idx_j):
        """
        Validate whether swapped and rattled positions are physically reasonable.
        ONLY checks for atomic overlap metrics using the Minimum Image Convention (MIC).
        """
        numbers = candidate.get_atomic_numbers()
        
        for i in (swap_idx_i, swap_idx_j):
            # Extract distances from atom 'i' to all other atoms using MIC
            for other_atom_idx in range(len(candidate)):
                if other_atom_idx in [swap_idx_i, swap_idx_j]:
                    continue
                
                covalent_dist = (
                    covalent_radii[numbers[i]] + 
                    covalent_radii[numbers[other_atom_idx]]
                )
                
                rmin = self.min_distance_scale * covalent_dist
                
                # Compute vector distance applying minimum image convention
                delta = new_positions[i] - candidate.positions[other_atom_idx]
                if candidate.pbc.any():
                    cell = candidate.get_cell()
                    from ase.geometry import find_mic
                    delta, _ = find_mic(delta, cell, pbc=candidate.pbc)
                
                dist = np.linalg.norm(delta)
                
                # Too close -> Overlap detected, reject swap configuration immediately
                if dist < rmin:
                    return False
                    
        return True

    def _get_candidates(self, candidate, parents, environment):
        if self.write_candidates_to_disk:
            write(f"candidate_start_{self.counter}.traj", candidate)
        
        n_template = len(candidate.get_template())
        all_numbers = candidate.get_atomic_numbers()
        
        # Indices of atoms that can actually be moved/swapped
        swappable_indices = np.array([
            i for i in range(n_template, len(candidate)) 
            if all_numbers[i] not in self.ignored_numbers
        ])
        
        swappable_numbers = np.unique(all_numbers[swappable_indices])
        if len(swappable_numbers) < 2:
            self.writer("AmorphPermutationGenerator: Need at least 2 unique species to swap.")
            return []
        
        number_of_swaps = self.max_number_of_swaps

        # Copy baseline coordinates
        new_positions = candidate.get_positions()
        backup_positions = candidate.get_positions()
        successful_swaps = 0

        for n in range(number_of_swaps):
            swap_found = False
            for _ in range(self.attempts):
                # Temporary coordinate buffer for this check optimization step
                iter_positions = new_positions.copy()

                # Select species and unique targets
                num_i = np.random.choice(swappable_numbers)
                remaining_numbers = swappable_numbers[swappable_numbers != num_i]
                num_j = np.random.choice(remaining_numbers)

                idx_i_list = [i for i in swappable_indices if all_numbers[i] == num_i]
                idx_j_list = [i for i in swappable_indices if all_numbers[i] == num_j]

                swap_idx_i = np.random.choice(idx_i_list)
                swap_idx_j = np.random.choice(idx_j_list)

                # Swap atomic positions 
                iter_positions[[swap_idx_i, swap_idx_j]] = iter_positions[[swap_idx_j, swap_idx_i]]

                # Apply positional displacements (rattling)
                if self.rattle_strength > 0:
                    for idx in (swap_idx_i, swap_idx_j):
                        if self.use_xy_only:
                            iter_positions[idx] += self.pos_add_disk(self.rattle_strength)
                        else:
                            iter_positions[idx] += self.pos_add_sphere(self.rattle_strength)

                if self.write_struct:
                    candidate.set_positions(iter_positions)
                    self.structure_counter += 1
                    filename = self.run_dir / f"permut_{self.structure_counter:04d}.xsf"
                    write(filename, candidate)
                    candidate.set_positions(backup_positions)
                    
                # physical distance environment parsing check (Overlap only)
                if self.check_overlap:
                    if not self.validate_swap(candidate, iter_positions, swap_idx_i, swap_idx_j):
                        continue
                
                new_positions = iter_positions
                swap_found = True
                successful_swaps += 1
                break
	
            if not swap_found:
                if self.print_result:
                    self.writer(f"Swap {n+1} failed after {self.attempts} attempts.")
            
        if successful_swaps == 0:
            return []
			
        candidate.set_positions(new_positions)
        
        if self.write_candidates_to_disk:
            write(f"candidate_final_{self.counter}.traj", candidate)
        
        return [candidate]
	
    def pos_add_disk(self, rattle_strength):
        r = rattle_strength * np.random.rand() ** (1/2)
        theta = np.random.uniform(low=0, high=2 * np.pi)
        return r * np.array([np.cos(theta), np.sin(theta), 0])
	
    def pos_add_sphere(self, rattle_strength):
        r = rattle_strength * np.random.rand() ** (1/3)
        theta = np.random.uniform(low=0, high=2 * np.pi)
        phi = np.random.uniform(low=0, high=np.pi)
        return r * np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])

    def _create_new_run_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{timestamp}"
        
        run_dir.mkdir(parents=True, exist_ok=True)
        self.structure_counter = 0
        
        print(f"Saved Permut to: {run_dir}")
        
        return run_dir
	
    def get_number_of_parents(self, sampler):
        return 1