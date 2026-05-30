import numpy as np
from ase import Atoms
from ase.io import write

from agox.generators.ABC_generator import GeneratorBaseClass
from agox.candidates import StandardCandidate, Candidate

from pathlib import Path
from datetime import datetime

from scipy.stats import maxwell

class HeteroStructRandomize(GeneratorBaseClass):
    """
    Generates heterostructures by depositing one atomic structure onto another (the template).
    Optionally applies random displacements ('rattling') to selected atoms to introduce variation.
    """
    name = "HeteroStructRandomize"
    __version__ = "0.0.1"

    def __init__(self,
                 slab_deposition: Atoms,
                 depos_pos_mean: bool = False,
                 write_struct: bool = False,
                 attempts: int = 100,
                 hetero_slab_dist: float = 2.3,
                 replace=True,
                 rattle_amplitude: int = 3,
                 n_rattle: int = 3,
                 generate_pristine: bool = False,
                 add_inter_layer_space: float = None,
                 boltzmann_rattle: bool = False,
                 output_dir: str = "generated_structures",
                 **kwargs):
        """
        Parameters:
        - slab_deposition: Atoms object to be deposited onto the template
        - depos_pos_mean: Whether to align deposition layer using its mean z-position (instead of min)
        - write_struct: If True, writes the final structure to an .xsf file
        - attempts: Maximum number of attempts per atom to find a valid displacement
        - hetero_slab_dist: Vertical separation between template and deposition slabs
        - replace: Whether to allow atom replacement in inherited structures
        - rattle_amplitude: Maximum displacement for rattling (in Å)
        - n_rattle: Expected number of atoms to apply rattling
        - generate_pristine: If True, skip all rattling and output pristine merged structure
        - add_inter_layer_space: Spacing apply between deposition layer (A)
        - boltzmann_rattle: If True, use Maxwell-distributed displacements (thermal-like)
        """
        super().__init__(replace=replace, **kwargs)
        self.attempts = attempts
        self.slab_deposition = slab_deposition
        self.hetero_slab_dist = hetero_slab_dist
        self.write_struct = write_struct
        self.rattle_amplitude = rattle_amplitude
        self.n_rattle = n_rattle
        self.generate_pristine = generate_pristine
        self.output_dir = Path(output_dir)
        self.structure_counter = 0
        self.run_dir = self._create_new_run_dir() if write_struct else None
        self.add_inter_layer_space = add_inter_layer_space
        self.boltzmann_rattle = boltzmann_rattle

    def _get_candidates(self,
                        candidate,
                        parents,
                        environment):
        """
        Creates a new heterostructure by stacking and optionally rattling atoms.
        """
        template = environment.get_template()

        # Align deposition structure above template
        template_pos_max = template.get_positions()[:, 2].max()
        slab_deposition = self.slab_deposition.copy()
        deposition_pos_min = slab_deposition.get_positions()[:, 2].min()
        dz = template_pos_max - deposition_pos_min + self.hetero_slab_dist
        slab_deposition.translate((0, 0, dz))

        # Optionally add spacing between monolayers in deposition slab
        if self.add_inter_layer_space:
            z_positions = slab_deposition.get_positions()[:, 2]
            tolerance = 0.3  # Å; max z-diff to group atoms into the same layer

            sorted_indices = np.argsort(z_positions)
            sorted_z = z_positions[sorted_indices]

            layers = []
            current_layer = [sorted_indices[0]]
            for i in range(1, len(sorted_indices)):
                if abs(sorted_z[i] - sorted_z[i - 1]) <= tolerance:
                    current_layer.append(sorted_indices[i])
                else:
                    layers.append(current_layer)
                    current_layer = [sorted_indices[i]]
            if current_layer:
                layers.append(current_layer)

            # Apply spacing between layers
            for layer_idx, atom_indices in enumerate(layers):
                if layer_idx == 0:
                    continue
                for atom_index in atom_indices:
                    slab_deposition.positions[atom_index, 2] += self.add_inter_layer_space * layer_idx

        # Append translated deposition atoms to the candidate structure
        candidate.extend(Atoms(numbers=slab_deposition.numbers,
                               positions=slab_deposition.get_positions()))

        # Apply rattling to selected atoms
        if not self.generate_pristine:
            indices_to_rattle = self.get_indices_to_rattle(candidate)
            for i in indices_to_rattle: # Iterate in the order returned by get_indices_to_rattle (top to bottom layers)
                for _ in range(self.attempts):
                    if self.boltzmann_rattle:
                        # Maxwell-distributed radius for thermal-like displacement
                        scale = self.rattle_amplitude / 3  # 99.7% of values within bound
                        radius = maxwell.rvs(scale=scale)
                        displacement = self.get_displacement_vector(radius)
                    else:
                        # Uniform-in-volume displacement
                        radius = self.rattle_amplitude * np.random.rand() ** (1 / self.get_dimensionality())
                        displacement = self.get_displacement_vector(radius)

                    suggested_position = candidate.positions[i] + displacement

                    if not self.check_confinement(suggested_position).all():
                        continue

                    if self.check_new_position(
                        candidate,
                        suggested_position,
                        candidate[i].number,
                        skipped_indices=[i],
                    ):
                        candidate[i].position = suggested_position
                        break

        # Optionally write the structure to file
        if self.write_struct:
            self.structure_counter += 1
            filename = self.run_dir / f"hetero_rand_{self.structure_counter:04d}.xsf"
            write(str(filename), candidate)

        return [candidate]

    def _create_new_run_dir(self):
        """Create a unique output subdirectory for this run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.structure_counter = 0
        print(f"[HeteroStructRandomize] Saving generated structures to: {run_dir}")
        return run_dir

    def get_indices_to_rattle(self, candidate: Candidate) -> np.ndarray:
        """
        Randomly select a subset of non-template atoms to rattle,
        grouped by layer and ordered from top to bottom.
        """
        template = candidate.get_template()
        n_template = len(template)
        n_total = len(candidate)
        n_non_template = n_total - n_template

        if n_non_template == 0:
            return np.array([], dtype=int)

        # Ordering the rattled monolayer so that it will rattle from the top layer.
        # Get indices of non-template atoms
        non_template_indices = np.arange(n_template, n_total)
        non_template_positions = candidate.positions[non_template_indices]
        non_template_z = non_template_positions[:, 2]

        # Group indices by layer based on z-coordinate
        tolerance = 0.3  # Å; max z-diff to group atoms into the same layer
        sorted_indices = np.argsort(non_template_z)[::-1] # Sort by z-height descending
        sorted_z = non_template_z[sorted_indices]

        layered_indices = []
        current_layer_indices = [non_template_indices[sorted_indices[0]]]

        for i in range(1, len(sorted_indices)):
            original_index = non_template_indices[sorted_indices[i]]
            if abs(sorted_z[i] - sorted_z[i-1]) <= tolerance:
                current_layer_indices.append(original_index)
            else:
                layered_indices.append(current_layer_indices)
                current_layer_indices = [original_index]

        if current_layer_indices:
            layered_indices.append(current_layer_indices)

        # Flatten the list of layers into a single array of indices, maintaining layer order
        rattle_indices = np.concatenate(layered_indices)

        # Select a subset if n_rattle is specified and less than total non-template atoms
        if self.n_rattle < n_non_template:
             # Randomly select n_rattle indices from the layered_indices while maintaining relative layer order
             # This is a simplified approach, a more robust method would involve
             # sampling within each layer proportionally or fixed number per layer.
             # For now, we'll just take the first n_rattle atoms in the layered list.
             # If you want to randomize within layers, you could shuffle each sublist in layered_indices
             # before concatenating.
             rattle_indices = rattle_indices[:self.n_rattle]

        return rattle_indices

    def get_number_of_parents(self, sampler):
        """This generator does not use parent candidates."""
        return 0

# # === Supercell Generation ===
# slab_mgofe_33 = slab_mgofe.repeat(supercell)
# write(f"slab_mgofe_33.xsf", slab_mgofe_33)

# # Repeat the MgO slab 3 times along x and y (to enlarge the surface), keep 1 layer along z
# slab_mgo = slab_mgo.repeat(supercell)
# write(f"slab_mgo_33.xsf", slab_mgo)

# # Repeat the Fe slab 3 times along x and y (to match the MgO slab size), keep 1 layer along z
# slab_fe = slab_fe.repeat(supercell)
# write(f"slab_fe_33.xsf", slab_fe)

# # === AGoX Environment ===
# slab_fe.pbc = [True, True, False]
# confinement_corner = np.array([0, 0, slab_fe.positions[:, 2].max() + dist_z_fe2o])

# environment = Environment(
#     template=slab_fe,
#     symbols=slab_mgo.get_chemical_formula(),
#     confinement_cell=slab_fe.cell.copy(),
#     confinement_corner=confinement_corner,
#     box_constraint_pbc=[True, True, False]
# )

# # === Structure Generators ===
# n_rattle = len(slab_mgo)

# hetero_generator = HeteroStructRandomize(
#                   **environment.get_confinement(),
#                   slab_deposition=slab_mgo,
#                   hetero_slab_dist=dist_z_fe2o,
#                   write_struct=True,
#                   rattle_amplitude=1.5,
#                   n_rattle=n_rattle,
#                   add_inter_layer_space=0.5,
#                   generate_pristine=False,
#               )

# hetero_candidate = hetero_generator(sampler=None, environment=environment)
# write('hetero_candidate.xsf', hetero_candidate)