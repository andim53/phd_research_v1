import numpy as np
from ase.io import write
from ase.data import covalent_radii, atomic_numbers
from agox.generators.ABC_generator import GeneratorBaseClass

class AmorphPermutationGenerator(GeneratorBaseClass):
	"""
    Permutes atoms of different species in a candidate structure with rattling.
    """

	name = "AmorphPermutationGenerator"
	
	def __init__(
        self,
        max_number_of_swaps=1,
        rattle_strength=0.0,
        use_xy_only=False,
        ignore_species=None,
        write_candidates_to_disk=False,
        replace=True,
        attempts=100,
		check_overlap=False, # check 70% covalent radii
		confinement_check = False,
        **kwargs,
    ):
		super().__init__(replace=replace, **kwargs)
		self.max_number_of_swaps = max_number_of_swaps
		self.rattle_strength = rattle_strength
		self.use_xy_only = use_xy_only
		self.write_candidates_to_disk = write_candidates_to_disk
		self.attempts = attempts
		self.check_overlap = check_overlap
		self.confinement_check = confinement_check
		
		self.ignored_numbers = []
		if ignore_species is not None:
			for s in ignore_species:
				if isinstance(s, str):
					self.ignored_numbers.append(atomic_numbers[s])
				else:
					self.ignored_numbers.append(s)
		
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
		
		cov_radii = covalent_radii[all_numbers]
		# number_of_swaps = np.random.randint(self.max_number_of_swaps) + 1
		number_of_swaps = self.max_number_of_swaps

		# start position
		new_positions = candidate.get_positions()
		successful_swaps = 0

		for n in range(number_of_swaps):
			swap_found = False
			for _ in range(self.attempts):
				# Temporary array for this specific attempt
				iter_positions = new_positions.copy()

				# select species and index
				num_i = np.random.choice(swappable_numbers)
				remaining_numbers = swappable_numbers[swappable_numbers != num_i]
				num_j = np.random.choice(remaining_numbers)

				idx_i_list = [i for i in swappable_indices if all_numbers[i] == num_i]
				idx_j_list = [i for i in swappable_indices if all_numbers[i] == num_j]

				swap_idx_i = np.random.choice(idx_i_list)
				swap_idx_j = np.random.choice(idx_j_list)

				# swap
				iter_positions[[swap_idx_i, swap_idx_j]] = iter_positions[[swap_idx_j, swap_idx_i]]

				if self.rattle_strength > 0:
					for i in swappable_indices:
						if self.use_xy_only:
							iter_positions[i] += self.pos_add_disk(self.rattle_strength)
						else:
							iter_positions[i] += self.pos_add_sphere(self.rattle_strength)
				
				if self.confinement_check:
					if not self.check_confinement(iter_positions).all():
							continue
				
				# check overlap
				if self.check_overlap:
					# Check if any atom in the whole system violates distance rules
					if not self._is_system_valid(iter_positions, cov_radii):
						continue
				
				new_positions = iter_positions
				swap_found = True
				successful_swaps += 1
				break
	
			if not swap_found:
				self.writer(f"Swap {n+1} failed after {self.attempts} attempts.")
		
		if successful_swaps == 0:
			return []
			
		candidate.set_positions(new_positions)
		
		if self.write_candidates_to_disk:
			write(f"candidate_final_{self.counter}.traj", candidate)
		
		return [candidate]

	def _is_system_valid(self, positions, radii):
		"""Checks the entire system for overlaps."""
		for i in range(len(positions)):
			dists = np.linalg.norm(positions[i+1:] - positions[i], axis=1)
			thresholds = 0.7 * (radii[i+1:] + radii[i])
			if np.any(dists < thresholds):
				return False
		return True
	
	def pos_add_disk(self, rattle_strength):
		r = rattle_strength * np.random.rand() ** (1/2)
		theta = np.random.uniform(low=0, high=2 * np.pi)
		return r * np.array([np.cos(theta), np.sin(theta), 0])
	
	def pos_add_sphere(self, rattle_strength):
		r = rattle_strength * np.random.rand() ** (1/3)
		theta = np.random.uniform(low=0, high=2 * np.pi)
		phi = np.random.uniform(low=0, high=np.pi)
		return r * np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])
	
	def get_number_of_parents(self, sampler):
		return 1