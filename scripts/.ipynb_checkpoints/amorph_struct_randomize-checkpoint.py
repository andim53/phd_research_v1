from agox.generators.ABC_generator import GeneratorBaseClass
from agox.candidates import Candidate

from pathlib import Path
from datetime import datetime

import numpy as np

from ase.io import write
from ase import Atoms
from ase.data import covalent_radii

class AmorphStructRandomize(GeneratorBaseClass):
    name = "AmorphStructRandomize"
    __version__ = "0.0.3"
    """
    0.0.3 Integrated covalent-radii-based distance validation loop.
    0.0.2 Clean up
    """

    def __init__(
        self, 
        amorph: "Atoms",
        
        write_struct: bool = True, # Save generated structures to disk
        write_temp_struct: bool = False, # save non valid struct
        
        attempts: int = 100, # Max attempts per atom to find valid displacement
        replace: bool = True,
        rattle_amplitude: float = 3.0, # Maximum displacement magnitude (Å)
        n_rattle: int = 3, # Unused here, reserved for future logic    
        generate_pristine: bool = False, # Skip rattling if True
        output_dir: str = "generated_structures", 
        
        check_covalent: bool = False,
        min_distance_scale: float = 0.8,  # Scale factor for covalent distance checks
        
        print_result: bool = False,
        **kwargs,
    ):
        super().__init__(replace=replace, **kwargs)
        self.amorph = amorph
        self.attempts = attempts
        
        self.write_struct = write_struct
        self.write_temp_struct = write_temp_struct
        
        self.rattle_amplitude = rattle_amplitude
        self.n_rattle = n_rattle
        self.generate_pristine = generate_pristine

        self.min_distance_scale = min_distance_scale
        self.check_covalent = check_covalent
        
        self.print_result = print_result
        self.output_dir = Path(output_dir)
        self.structure_counter = 0
        self.run_dir = self._create_new_run_dir() if write_struct else None

    def check_too_close(self, atoms: Atoms, scale: float = None) -> bool:
        """
        Check whether any atom pair is too close using covalent radii 
        and the minimum image convention (MIC).
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
                    covalent_radii[numbers[i]] + covalent_radii[numbers[j]]
                )
                if dmat[i, j] < cutoff:
                    return True
        return False
        
    def _get_candidates(self, candidate, parents, environment):        
        # Initialize candidate with amorphous template properties
        candidate.extend(
            Atoms(
                numbers=self.amorph.numbers,
                positions=self.amorph.get_positions(),
                # cell=self.amorph.get_cell(),
                # pbc=self.amorph.get_pbc()
            )
        )
        
        if not self.generate_pristine:
            indices_to_rattle = self.get_indices_to_rattle(candidate)
            
            for i in indices_to_rattle:
                valid_displacement_found = False
                original_position = candidate.positions[i].copy()

                for _ in range(self.attempts):
                    radius = (
                        self.rattle_amplitude
                        * np.random.rand() ** (1 / self.get_dimensionality())
                    )
                    
                    displacement = self.get_displacement_vector(radius)
                    new_pos = original_position + displacement

                    # Check confinement limits:                       
                    if not self.check_confinement(new_pos).all():
                        continue
                        
                    # Check that suggested_position is not too close/far to/from other atoms
                    # Skips the atom it self.
                    if not self.check_new_position(
                        candidate,
                        new_pos,
                        candidate[i].number,
                        skipped_indices=[i],
                    ):
                        continue

                    # Temporarily assign position to evaluate physical distance overlaps
                    candidate[i].position = new_pos

                    if self.write_temp_struct and self.write_struct:
                        if self.write_struct:
                            self.structure_counter += 1
                            filename = self.run_dir / f"temp_amorph_rand_{self.structure_counter:04d}.xsf"
                            write(filename, candidate)

                    # New Covalent Distance Validation Check
                    if self.check_covalent:
                        if self.check_too_close(candidate):
                            # Revert and try a new random vector if too close
                            # candidate[i].position = original_position
                            continue
                    
                    # If it passes all checks, keep the position and stop attempting for this atom
                    valid_displacement_found = True
                    break
                
                # # If a valid spot wasn't found, it defaults back to its original position
                # if not valid_displacement_found and self.print_result:
                #     print(f"Warning: Atom {i} could not find a valid displacement within {self.attempts} attempts.")
        
        # Final structural sanity check before returning
        if self.check_covalent and self.check_too_close(candidate):
            if self.print_result:
                print("Rejected candidate structure: Total structure failed distance validation.")
            return []

        if self.write_struct and not self.write_temp_struct:
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