import os
import numpy as np

from agox import AGOX
from agox.environments import Environment
from agox.generators import RattleGenerator
from agox.databases import Database
from agox.models.descriptors.fingerprint import Fingerprint
from agox.models.GPR import GPR
from agox.models.GPR.kernels import RBF, Noise, Constant as C
from agox.models.GPR.priors import Repulsive
from agox.samplers import KMeansSampler
from agox.collectors import ParallelCollector, StandardCollector
from agox.acquisitors import LowerConfidenceBoundAcquisitor
from agox.postprocessors import ParallelRelaxPostprocess, RelaxPostprocess
from agox.helpers import SubprocessGPAW
from agox.evaluators import LocalOptimizationEvaluator
from agox.samplers import FixedSampler

from ase import Atoms
from ase.constraints import FixAtoms
from ase.build import surface, bulk
from ase.io import read, write

from scripts.build_mgo_stack import build_mgo_stack
from scripts.build_fe_stack import build_fe_stack
from scripts.hetero_struct_randomize import HeteroStructRandomize
from scripts.plot_structure import plot_structure
from scripts.build_heteroStruct import build_heteroStruct
from scripts.remove_random_atoms_by_species import remove_random_atoms_by_species
from scripts.add_adsorbate_to_hollows import add_adsorbate_to_hollows

vacuum = 20
a_mgo = 4.212
a_fe = 2.870190   # Fe optimized lcao lattice constant (Å)

a_mgo_matched = a_mgo / np.sqrt(2)
strain = (a_mgo_matched - a_fe) / a_fe * 100

"""
Control Strain
0.0 = Fe lattice
1.0 = Fe stretch to fit MgO

Simul: 
0, 0.25, 0.5, 0.75, 1
"""
interpolation_factor = 0
a_custom = a_fe + interpolation_factor * (a_mgo_matched - a_fe)

dist_z_fe2o = 0.5 # experimental: 2.3 A
ncores = 24 #24; 16 cores for genkai
supercell = (5 , 5, 1)
kpts = (1, 1, 1)

kappa=2
N_iterations = 100

mgo_layer_number = 1
fe_layer_number = 1
confinement_cell_height_multiplyer = 4 # multiply env cell height

"""
Concenstration controls. 
Removing atoms.
5x5 1 monolayer is 25 atoms (Fe layer). Remove 25 remove one monolayer.
"""
removed_num = 0 

num_candidates={0:[20,0], 10:[10,10], 25:[0,20]}
sample_size = 20

"""
Atom adition
"""
symbol_add = 'B'
z_height_add = 0 # A
num_atoms_add = 2

for seed in range(103):
	print(F"Start seed: {seed}")
	
	path_result = f"seed_{seed}/0_result"
	path_xsf = f"{path_result}/0_xsf"
	path_fig = f"{path_result}/1_fig"
	db_dir = f"seed_{seed}/1_db"
	latt_log = f'{path_result}/latt_log.md'

	for d in [path_xsf, path_fig, db_dir]:
		os.makedirs(d, exist_ok=True)
		
	with open(latt_log, 'w') as f:
		f.write(f"{a_fe=}\n{a_mgo=}\n{a_mgo_matched=}\n{strain=:.2f}%\n")
		
	bulk_mgo = bulk('MgO', 'rocksalt', a=a_mgo, cubic=True)
	slab_mgo = surface(bulk_mgo, (0,0,1), layers=1, vacuum=vacuum)

	"""
	Calculating distance between MgO layer (Top Mg and bottom O).
	"""
	z_positions = slab_mgo.get_positions()[:, 2]
	unique_z = np.unique(np.round(z_positions, 5))
	
	if len(unique_z) >= 2:
		unique_z.sort()
		dist_mgo = unique_z[1] - unique_z[0]
	
	fe_bulk = bulk('Fe', 'bcc', a=a_custom, cubic=True)
	slab_fe_base = surface(fe_bulk, (0, 0, 1), layers=1, vacuum=vacuum)
	
	slab_mgofe = build_mgo_stack(slab_fe_base, num_layers=mgo_layer_number, dist_mgo=dist_mgo, vacuum=vacuum, output_path=f"{path_xsf}/slab_mgofe.xsf")
	slab_mgo = slab_mgofe[[atom.symbol != 'Fe' for atom in slab_mgofe]].repeat(supercell)
	slab_fe = build_fe_stack(slab_fe_base, num_layers=fe_layer_number, vacuum=vacuum, output_path=f"{path_xsf}/slab_fe.xsf").repeat(supercell)
	
	slab_deposition = slab_fe.copy() 
	slab_substrate = slab_mgo.copy()
    
	# Consentrations controls
	slab_deposition = remove_random_atoms_by_species(slab_deposition, 'Fe', removed_num)

	z_coordinates = slab_deposition.positions[:, 2]
	bottom_height = np.min(z_coordinates)

	# Boron controls
	slab_deposition = add_adsorbate_to_hollows(slab_deposition, symbol=symbol_add,height=z_height_add, num_atoms=num_atoms_add,seed=seed)

	# Build test
	test_heteroStruct, test_substrate, test_deposition, substrate_layer_heights, deposition_layer_heights = build_heteroStruct(slab_substrate, slab_deposition, output_path=f'{path_xsf}/heteroStruct.xsf')

	slab_substrate.pbc = [True, True, False]
	confinement_corner = np.array([0, 0, slab_substrate.positions[:, 2].max() + dist_z_fe2o])
	
	z_pos = slab_deposition.get_positions()[:, 2]
	h_dep = max(z_pos.max() - z_pos.min(), 2.1)
	confinement_cell = slab_deposition.cell.copy()
	confinement_cell[2, 2] = h_dep * confinement_cell_height_multiplyer

	environment = Environment(
		template=slab_substrate,
		symbols=slab_deposition.get_chemical_formula(),
		confinement_cell=confinement_cell,
		confinement_corner=confinement_corner,
		box_constraint_pbc=[True, True, False]
	)
	
	n_rattle = len(slab_deposition)
	generators = [
		HeteroStructRandomize(
			**environment.get_confinement(),
			slab_deposition=slab_deposition,
			hetero_slab_dist=dist_z_fe2o,
			rattle_amplitude=1.5,
			n_rattle=n_rattle,
			generate_pristine=False,
			write_struct=True,
		),
		RattleGenerator(
			**environment.get_confinement(),
			n_rattle=int(n_rattle * 0.5),
			rattle_amplitude=2.3
		),
	]
	
	hetero_candidate = generators[0](sampler=None, environment=environment)[0]
	write(f'{path_xsf}/hetero_candidate.xsf', hetero_candidate)
	
	sampler = FixedSampler(hetero_candidate)
	rattle_candidate = generators[1](sampler, environment)[0]
	write(f'{path_xsf}/rattle_candidate.xsf', rattle_candidate)
	
	database = Database(filename=f"{db_dir}/db_{seed}.db", order=5)
	descriptor = Fingerprint(environment=environment)
	
	beta = 0.01
	kernel = C(5000, (1, 1e5)) * (C(beta, (beta, beta)) * RBF() + C(1-beta, (1-beta, 1-beta)) * RBF()) + Noise(0.01, (0.01, 0.01))
	model = GPR(descriptor=descriptor, kernel=kernel, database=database, prior=Repulsive())
	
	sampler = KMeansSampler(descriptor=descriptor, database=database, sample_size=sample_size)
	collector = ParallelCollector(
		generators=generators,
		sampler=sampler,
		environment=environment,
		num_candidates=num_candidates,
		order=1
	)
	
	acquisitor = LowerConfidenceBoundAcquisitor(model=model, kappa=kappa, order=3)
	
	relaxer = ParallelRelaxPostprocess(
		model=acquisitor.get_acquisition_calculator(),
		constraints=environment.get_constraints(),
		optimizer_run_kwargs={"steps": 100},
		start_relax=10,
		order=2
	)
	
	calc = SubprocessGPAW(
		ncores=ncores,
		mode={"name": "lcao"},
		basis="dzp",
		xc="PBE",
		mixer={"backend": "pulay", "beta": 0.05, "nmaxold": 5, "weight": 100},
		convergence={"energy": 1e-4, "density": 1e-3, "eigenstates": 1e-3},
		txt=f"output_seed_{seed}.txt",
		kpts=kpts,
		symmetry='off',
		nbands='nao',
		maxiter=100,
		occupations={"name": "fermi-dirac", "width": 0.05},
		hund=True,
		spinpol=True
	)
	
	evaluator = LocalOptimizationEvaluator(
		calc,
		gets={"get_key": "prioritized_candidates"},
		optimizer_run_kwargs={"fmax": 0.05, "steps": 1},
		constraints=environment.get_constraints(),
		store_trajectory=False,
		order=4
	)
	
	agox = AGOX(collector, relaxer, acquisitor, evaluator, database, seed=seed)
	agox.run(N_iterations=N_iterations)