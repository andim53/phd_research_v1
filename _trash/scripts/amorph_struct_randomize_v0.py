from agox.generators.ABC_generator import GeneratorBaseClass
from agox.candidates import Candidate

from pathlib import Path
from datetime import datetime

import numpy as np

from ase.io import write
from ase import Atoms

class AmorphStructRandomize(GeneratorBaseClass):
	name = "AmorphStructRandomize"
	__version__ = "0.0.2"
	"""
	0.0.2 Clean up
	"""
	
	def __init__(
		self, 
		amorph: Atoms,
		write_struct: bool = False, # Save generated structures to disk
		attempts: int = 100, # Max attempts per atom to find valid displacement
		replace: bool = True,
		rattle_amplitude: float = 3.0, # Maximum displacement magnitude (Å)
		n_rattle: int = 3, # Unused here, reserved for future logic	
		generate_pristine: bool = False, # Skip rattling if True
		output_dir: str = "generated_structures",
		**kwargs,
	):
	
		super().__init__(replace=replace, **kwargs)
		self.amorph = amorph
		self.attempts = attempts
		self.write_struct = write_struct
		self.rattle_amplitude = rattle_amplitude
		self.n_rattle = n_rattle
		self.generate_pristine = generate_pristine
		
		self.output_dir = Path(output_dir)
		self.structure_counter = 0
		self.run_dir = self._create_new_run_dir() if write_struct else None
	
	def _get_candidates(self, candidate, parents, environment):		
		candidate.extend(
			Atoms(
				numbers=self.amorph.numbers,
				positions=self.amorph.get_positions()
			)
		)
		
		if not self.generate_pristine:
			indices_to_rattle = self.get_indices_to_rattle(candidate)
			for i in indices_to_rattle:
				for _ in range(self.attempts):
					radius = (
						self.rattle_amplitude
						* np.random.rand() ** (1 / self.get_dimensionality())
					)
					
					displacement = self.get_displacement_vector(radius)
					new_pos = candidate.positions[i] + displacement
					
					if not self.check_confinement(new_pos).all():
						continue
					
					if self.check_new_position(
						candidate,
						new_pos,
						candidate[i].number,
						skipped_indices=[i],
					):
						candidate[i].position = new_pos
						break
		
		if self.write_struct:
			self.structure_counter += 1
			filename = self.run_dir / f"amorph_rand_{self.structure_counter:04d}.xsf"
			write(filename, candidate)
		
		return [candidate]
		
	
	def _create_new_run_dir(self):
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		run_dir = self.output_dir / f"run_{timestamp}"
		
		run_dir.mkdir(parents=True, exist_ok=True)
		self.structure_counter = 0
		
		print(f"Saved Amorph to: {run_dir}")
		
		return run_dir
	
	def get_indices_to_rattle(self, candidate: Candidate) -> np.ndarray:
		return np.arange(len(self.amorph))
		
	def get_number_of_parents(self, sampler):
		return 0