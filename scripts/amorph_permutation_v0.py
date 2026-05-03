import numpy as np
from ase.io import write

from agox.generators.ABC_generator import GeneratorBaseClass
from agox.candidates import Candidate

class AmorphPermutationGenerator(GeneratorBaseClass):

	"""
	No check covalent distance. Just change positions. 
	"""
	
	name = "AmorphPermutationGenerator"
	
	def __init__(
		self,
		max_number_of_swaps=1,
		rattle_strength=0.0,
		use_xy_only=False,
		ignore_species = None,
		write_candidates_to_disk=False,
		replace=True,
		attempts = 100,
		**kwargs,
	):
		super().__init__(replace=replace, **kwargs)
		self.max_number_of_swaps = max_number_of_swaps
		self.rattle_strength = rattle_strength
		self.use_xy_only = use_xy_only
		self.write_candidates_to_disk = write_candidates_to_disk
		self.attempts = attempts
		
		if ignore_species is not None:
			self.ignored_numbers = [ase_symbols_converters(s).numbers[0] for s in ignored_species]
		else:
			self.ignored_numbers = []
		
	
	def _get_candidates(self, candidate, parents, environment):
		if self.write_candidates_to_disk:
			write(f"candidate_{self.counter}.traj", candidate)
		
		number_of_template_atoms = len(candidate.get_template())
		all_numbers = candidate.get_atomic_numbers()
		
		swappable_indices = [
			i for i in range(number_of_template_atoms, len(candidate))
			if all_numbers[i] not in self.ignored_numbers
		]
		
		swappable_numbers = np.unique(all_numbers[swappable_indices])
		assert len(swappable_numbers) > 1, "At least 2 species"
		
		new_positions = candidate.get_positions()
		number_of_swaps = np.random.randint(self.max_number_of_swaps) + 1
		
		for n in range(number_of_swaps):
			
			for _ in range(self.attempts):
				
				num_i = np.random.choice(swappable_numbers)
				remaining_numbers = swappable_numbers[swappable_numbers != num_i]
				num_j = np.random.choice(remaining_numbers)
				
				idx_i = [i for i in swappable_indices if all_numbers[i] == num_i]
				idx_j = [i for i in swappable_indices if all_numbers[i] == num_j]
				
				swap_idx_i = np.random.choice(idx_i)
				swap_idx_j = np.random.choice(idx_j)
				
				new_positions[[swap_idx_i, swap_idx_j]] = new_positions[[swap_idx_j, swap_idx_i]]
				
			
				if self.rattle_strength > 0:
					for i in swappable_indices:
						if self.use_xy_only:
							new_positions[i] += self.pos_add_disk(self.rattle_strength)
						else:
							new_positions[i] += self.pos_add_sphere(self.rattle_strength)
				
				if not self.check_confinement(new_positions).all():
						continue
					
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
	
	def get_number_of_parents(self, sampler):
		return 1