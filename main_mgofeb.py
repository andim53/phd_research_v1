import os
import re
import numpy as np

# ASE (Atomic Simulation Environment) imports
from ase import Atoms
from ase.build import bulk, surface
from ase.constraints import FixAtoms
from ase.io import read, write

# AGOX infrastructure & workflow imports
from agox import AGOX
from agox.environments import Environment
from agox.databases import Database
from agox.helpers import SubprocessGPAW

# AGOX pipeline components
from agox.collectors import ParallelCollector, StandardCollector
from agox.acquisitors import LowerConfidenceBoundAcquisitor
from agox.evaluators import LocalOptimizationEvaluator
from agox.postprocessors import ParallelRelaxPostprocess, RelaxPostprocess
from agox.samplers import FixedSampler, KMeansSampler

# AGOX Candidate Generators
from agox.generators import RattleGenerator # , PermutationGenerator

# AGOX Machine Learning & Surrogate Model imports
from agox.models.GPR import GPR
from agox.models.GPR.kernels import RBF, Noise, Constant as C
from agox.models.GPR.priors import Repulsive
from agox.models.descriptors.fingerprint import Fingerprint

# Custom project modules for structural manipulation
from scripts.build_fe_stack import build_fe_stack
from scripts.build_mgo_stack import build_mgo_stack
from scripts.build_heteroStruct import build_heteroStruct
from scripts.hetero_struct_randomize import HeteroStructRandomize
from scripts.remove_random_atoms_by_species import remove_random_atoms_by_species
from scripts.add_adsorbate_to_hollows import add_adsorbate_to_hollows
from scripts.plot_structure import plot_structure
from scripts.global_permutation_generator import GlobalPermutationGenerator

# =============================================================================
# CORE CONFIGURATION & INTERFACE GEOMETRY
# =============================================================================

# Substrate properties
substrate_species = 'MgO'
substrate_crystal_structure = 'rocksalt'
a_substrate = 4.212
substrate_layer_number = 1

# Deposition film properties
deposition_species = 'Fe'
deposition_crystal_structure = 'bcc'
a_deposition = 2.870190  # Optimized LCAO lattice constant (Å)
deposition_layer_number = 1

# Interface & cell dimensions
vacuum = 20
dist_z_interface = 0.5   # Initial vertical layer separation (Å)
supercell = (5, 5, 1)    # Lateral repetition matrix
ncores = 24              # CPU cores assigned per GPAW instance
kpts = (1, 1, 1)         # Gamma-point Brillouin zone sampling

# Lattice matching & strain calculation
# (0.0 = Pure deposition lattice, 1.0 = Coherent with substrate)
interpolation_factor = 0
a_substrate_matched = a_substrate / np.sqrt(2)
strain = (a_substrate_matched - a_deposition) / a_deposition * 100
a_custom = a_deposition + interpolation_factor * (a_substrate_matched - a_deposition)

# =============================================================================
# AGOX ACTIVE LEARNING HYPERPARAMETERS
# =============================================================================
kappa = 2                               # LCB exploration/exploitation weight
N_iterations = 100                      # Max search iterations
sample_size = 20                        # KMeans clustering pool capacity
confinement_cell_height_multiplyer = 4  # Search space Z-box multiplier

# Generator population strategy per iteration window: [Randomize, Rattle, Permute]
num_candidates = {
    0: [20, 0, 0],    # Stage: Global structure exploration
    10: [10, 5, 5],   # Stage: Mixed search landscape
    25: [0, 10, 10]   # Stage: Fine-tuning local minima near hulls
}

# =============================================================================
# CHEMICAL DEFECTS & ADSORBATE DOPING
# =============================================================================
removed_num = 0       # Number of deposition film atoms to prune for vacancy defects
symbol_add = 'B'      # Dopant/adsorbate atomic species
z_height_add = 0      # Vertical adsorption height offset from the surface plane
num_atoms_add = 7    #25Fe; 21.87%:7; 30.55%:11; 40.47%:17# Number of adsorbate atoms to insert into available hollows

# =============================================================================
# ACTIVE LEARNING EXECUTION LOOP
# =============================================================================
for seed in range(103):
    print(f"\n==================================================")
    print(f"Executing Global Optimization Sequence | Seed: {seed}")
    print(f"==================================================")
    
    # Establish directory routing paths
    path_result = f"seed_{seed}/0_result"
    path_xsf = f"{path_result}/0_xsf"
    path_fig = f"{path_result}/1_fig"
    db_dir = f"seed_{seed}/1_db"
    latt_log = f'{path_result}/latt_log.md'

    for d in [path_xsf, path_fig, db_dir]:
        os.makedirs(d, exist_ok=True)
        
    # Write geometry and strain reports
    with open(latt_log, 'w') as f:
        f.write(f"{a_deposition=}\n{a_substrate=}\n{a_substrate_matched=}\n{strain=:.2f}%\n")
        
    # --- Substrate Slab Construction ---
    bulk_substrate = bulk(substrate_species, substrate_crystal_structure, a=a_substrate, cubic=True)
    slab_substrate_initial = surface(bulk_substrate, (0, 0, 1), layers=1, vacuum=vacuum)

    # Compute vertical layer spacing within the substrate
    z_positions = slab_substrate_initial.get_positions()[:, 2]
    unique_z = np.unique(np.round(z_positions, 5))
    if len(unique_z) >= 2:
        unique_z.sort()
        dist_substrate = unique_z[1] - unique_z[0]
    else:
        dist_substrate = 2.106  # Empirical fallback default
    
    # --- Deposition Slab Construction ---
    bulk_deposition = bulk(deposition_species, deposition_crystal_structure, a=a_custom, cubic=True)
    slab_deposition_initial = surface(bulk_deposition, (0, 0, 1), layers=1, vacuum=vacuum)
    
    # Assemble individual component stacks
    slab_combined_stack = build_mgo_stack(
        slab_deposition_initial, num_layers=substrate_layer_number, dist_mgo=dist_substrate, 
        vacuum=vacuum, output_path=f"{path_xsf}/slab_combined_stack.xsf", rotate_system=False
    )
    slab_substrate_base = slab_combined_stack[[atom.symbol != deposition_species for atom in slab_combined_stack]].repeat(supercell)
    slab_deposition_base = build_fe_stack(
        slab_deposition_initial, num_layers=deposition_layer_number, vacuum=vacuum, 
        output_path=f"{path_xsf}/slab_deposition.xsf"
    ).repeat(supercell)
    
    slab_deposition = slab_deposition_base.copy() 
    slab_substrate = slab_substrate_base.copy()
    
    # --- Structural Mutations (Vacancies & Adsorption) ---
    if removed_num > 0:
        slab_deposition = remove_random_atoms_by_species(slab_deposition, deposition_species, removed_num)

    # Dynamic hollow site detection and addition
    slab_deposition = add_adsorbate_to_hollows(
        slab_deposition, symbol=symbol_add, height=z_height_add, 
        num_atoms=num_atoms_add, seed=seed
    )

    # Write full trial hetero-interface
    test_hetero, _, _, _, _ = build_heteroStruct(
        slab_substrate, slab_deposition, output_path=f'{path_xsf}/heteroStruct.xsf'
    )

    # --- AGOX Search Boundary Setup ---
    slab_substrate.pbc = [True, True, False]
    confinement_corner = np.array([0, 0, slab_substrate.positions[:, 2].max() + dist_z_interface])
    
    # Establish bounding search volumes based on deposition cluster height
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
    
    # --- Candidate Generators Setup ---
    n_rattle = len(slab_deposition)
    generators = [
        # Large-scale stochastic structural layout
        HeteroStructRandomize(
            **environment.get_confinement(),
            slab_deposition=slab_deposition,
            hetero_slab_dist=dist_z_interface,
            rattle_amplitude=1.5,
            n_rattle=n_rattle,
            generate_pristine=False,
            write_struct=True,
        ),
        # Local atomic coordinate rattle adjustments
        RattleGenerator(
            **environment.get_confinement(),
            n_rattle=int(n_rattle * 0.5),
            rattle_amplitude=2.3
        ),
        # Discrete elemental permutation swapping
        GlobalPermutationGenerator(
            **environment.get_confinement(),
            max_number_of_swaps=num_atoms_add,
            rattle_strength=0.5,
        )
    ]
    
    # Validate and write out generator baseline states
    try:
        hetero_candidate = generators[0](sampler=None, environment=environment)[0]
        write(f'{path_xsf}/hetero_candidate.xsf', hetero_candidate)
        
        sampler_fixed = FixedSampler(hetero_candidate)
        rattle_candidate = generators[1](sampler_fixed, environment)[0]
        write(f'{path_xsf}/rattle_candidate.xsf', rattle_candidate)
    
        permutation_candidate = generators[2](sampler_fixed, environment)[0]
        write(f'{path_xsf}/permut_candidate.xsf', permutation_candidate)
        
    except Exception as e:
        print(f"--> [ERROR] Candidate generation failed at Seed {seed}: {e}")
    
    # --- Machine Learning Models & Database Storage ---
    database = Database(filename=f"{db_dir}/db_{seed}.db", order=5)
    descriptor = Fingerprint(environment=environment)
    
    # Compound Gaussian Process Regression (GPR) kernel definition
    beta = 0.01
    kernel = C(5000, (1, 1e5)) * (
        C(beta, (beta, beta)) * RBF() + C(1 - beta, (1 - beta, 1 - beta)) * RBF()
    ) + Noise(0.01, (0.01, 0.01))
    
    model = GPR(descriptor=descriptor, kernel=kernel, database=database, prior=Repulsive())
    
    # Candidate acquisition and parallel clustering pool configuration
    sampler_kmeans = KMeansSampler(descriptor=descriptor, database=database, sample_size=sample_size)
    collector = ParallelCollector(
        generators=generators,
        sampler=sampler_kmeans,
        environment=environment,
        num_candidates=num_candidates,
        order=1
    )
    
    # --- Evaluators & Quantum Calculators Setup (GPAW backend) ---
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
    
    # --- AGOX Engine Orchestration ---
    agox = AGOX(collector, relaxer, acquisitor, evaluator, database, seed=seed)
    
    # To run the core global optimization search loop, uncomment the following line:
    # agox.run(N_iterations=N_iterations)