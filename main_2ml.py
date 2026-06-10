"""
AGOX Workflow Script for Fe/MgO interface System
- Constructs matched Fe/MgO interface
- Configures AGOX optimization loop with GPAW
"""

import os
import numpy as np
from ase import Atoms
from ase.build import surface, bulk
from ase.io import read, write
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
from agox import AGOX
from agox.samplers import FixedSampler
from ase.constraints import FixAtoms
from ase import Atoms

# from icecream import ic

from scripts.build_mgo_stack import build_mgo_stack
from scripts.build_fe_stack import build_fe_stack
from scripts.hetero_struct_randomize import HeteroStructRandomize
from scripts.plot_structure import plot_structure
from scripts.build_heteroStruct import build_heteroStruct

# ==== Parameters ====
vacuum = 20               # Vacuum spacing in Å
a_mgo = 4.212             # MgO bulk lattice constant (Å)
a_fe = 2.870190           # Fe optimized lcao lattice constant (Å)
dist_z_fe2o = 0.5 #2.3         # Interface spacing (Å), experimental
ncores = 24 #24 # 16 cores for genkai
supercell = (5, 5, 1)
kpts = (1, 1, 1)
kappa=2
N_iterations = 100

# ==== Slab Number of Layer
mgo_layer_number = 2
fe_layer_number = 1
confinement_cell_height_multiplyer = 4

# ==== Generators Parameters
num_candidates={0:[20,0], 10:[10,10], 25:[0,20]}
sample_size = 20

# ==== Iteration Loop: Seed 5 to 100 ====
for seed in range(3, 105):
    print(f"--- Starting AGOX Run with Seed: {seed} ---")

    # Path setup per seed
    path_result = f"seed_{seed}/0_result"
    path_xsf = f"{path_result}/0_xsf"
    path_fig = f"{path_result}/1_fig"
    db_dir = f"seed_{seed}/1_db"
    latt_log = f'{path_result}/latt_log.md'

    for d in [path_xsf, path_fig, db_dir]:
        os.makedirs(d, exist_ok=True)

    # === Lattice Matching & Base Slabs ===
    a_mgo_matched = a_mgo / np.sqrt(2)
    strain = (a_mgo_matched - a_fe) / a_fe * 100

    with open(latt_log, 'w') as f:
        f.write(f"{a_fe=}\n{a_mgo=}\n{a_mgo_matched=}\n{strain=:.2f}%\n")
    
    bulk_mgo = bulk('MgO', 'rocksalt', a=a_mgo, cubic=True)
    slab_mgo = surface(bulk_mgo, (0, 0, 1), layers=1, vacuum=vacuum)
    slab_fe_base = surface('Fe', (0, 0, 1), layers=1, vacuum=vacuum)

    # === Build Stacks ===
    slab_mgofe = build_mgo_stack(slab_fe_base, num_layers=mgo_layer_number, vacuum=vacuum, output_path=f"{path_xsf}/slab_mgofe.xsf", rotate_system=True)
    slab_mgo = slab_mgofe[[atom.symbol != 'Fe' for atom in slab_mgofe]].repeat(supercell)
    write(f"{path_xsf}/slab_mgo.xsf", slab_mgo)
    slab_fe = build_fe_stack(slab_fe_base, num_layers=fe_layer_number, vacuum=vacuum, output_path=f"{path_xsf}/slab_fe.xsf").repeat(supercell)

    slab_substrate = slab_mgo.copy()
    write(f"{path_xsf}/slab_substrate.xsf", slab_substrate)
    slab_deposition = slab_fe.copy()
    write(f"{path_xsf}/slab_deposition.xsf", slab_deposition)
    build_heteroStruct(slab_substrate, slab_deposition, output_path=f'{path_xsf}/heteroStruct.xsf')

    # === Environment ===
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

    # === Generators ===
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

    try:
        hetero_candidate = generators[0](sampler=None, environment=environment)[0]
        write(f'{path_xsf}/hetero_candidate.xsf', hetero_candidate)
        
        sampler = FixedSampler(hetero_candidate)
        rattle_candidate = generators[1](sampler, environment)[0]
        write(f'{path_xsf}/rattle_candidate.xsf', rattle_candidate)
        
    except Exception:
        print('fail generate candidate')

    # === Database & Model ===
    database = Database(filename=f"{db_dir}/db_{seed}.db", order=5)
    descriptor = Fingerprint(environment=environment) #oganov

    beta = 0.01
    kernel = C(5000, (1, 1e5)) * (C(beta, (beta, beta)) * RBF() + C(1-beta, (1-beta, 1-beta)) * RBF()) + Noise(0.01, (0.01, 0.01))
    model = GPR(descriptor=descriptor, kernel=kernel, database=database, prior=Repulsive())

    # === Collector & Acquisitor ===
    sampler = KMeansSampler(descriptor=descriptor, database=database, sample_size=sample_size)
    
    collector = ParallelCollector(
        generators=generators, 
        sampler=sampler, 
        environment=environment, 
        num_candidates=num_candidates, 
        order=1)

    acquisitor = LowerConfidenceBoundAcquisitor(model=model, kappa=kappa, order=3) # LCB

    # === Relaxer & Evaluator ===
    relaxer = ParallelRelaxPostprocess(
        model=acquisitor.get_acquisition_calculator(), 
        constraints=environment.get_constraints(), 
        optimizer_run_kwargs={"steps": 100}, 
        start_relax=10, 
        order=2)

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
        order=4)

    # === Run AGOX ===
    agox = AGOX(collector, relaxer, acquisitor, evaluator, database, seed=seed)
    agox.run(N_iterations=N_iterations)

    # === Test Generators ===
    # hetero_candidate = generators[0](sampler=None, environment=environment)[0]
    # write(f'{path_xsf}/hetero_candidate.xsf', hetero_candidate)

    # sampler = FixedSampler(hetero_candidate)
    # rattle_candidate = generators[1](sampler, environment)[0]
    # write(f'{path_xsf}/rattle_candidate.xsf', rattle_candidate)

    # === Plotting Test Generators
    # cell_offset = np.array([10.0, 10.0, 0.0]) # Example offset
    # custom_colors = {'O': 'red', 'Mg': 'orange', 'Fe': 'green'}
    # for label, i in [ ('hetero', hetero_candidate), ('rattle', rattle_candidate)]:
    #     # Apply constraints for visualization
    #     unique_substrate_symbols = list(set(slab_substrate.get_chemical_symbols()))
    #     substrate_indices = [atom.index for atom in i if (atom.symbol in unique_substrate_symbols) and (round(atom.position[2], 2) in substrate_layer)]
    #     constraint = FixAtoms(indices=substrate_indices)
    #     i.set_constraint(constraint)
    #     plot_structure(
    #             i,
    #             plane='yz+',
    #             save_path=f"{path_fig}/figure_{label}.png",
    #             environment=environment,
    #             figsize = (20,20),
    #             repeat=5,
    #             cell_offset = cell_offset,
    #             linewidths_cell = 5,
    #             linewidths_environment = 5,
    #             plot_show = False,
    #     )


