import os
import numpy as np

from ase import Atoms
from ase.build import bulk
from ase.io import write

from agox import AGOX
from agox.environments import Environment
from agox.generators import RattleGenerator
from agox.samplers import KMeansSampler, FixedSampler
from agox.collectors import ParallelCollector, StandardCollector
from agox.acquisitors import LowerConfidenceBoundAcquisitor
from agox.postprocessors import ParallelRelaxPostprocess
from agox.evaluators import LocalOptimizationEvaluator
from agox.databases import Database
from agox.models.descriptors.fingerprint import Fingerprint
from agox.models.descriptors.soap import SOAP

from agox.models.GPR.kernels import RBF, Noise, Constant as C
from agox.models.GPR.priors import Repulsive
from agox.helpers import SubprocessGPAW

# from agox.models.GPR import GPR
from agox.models.GPR import SparseGPR
from agox.models.GPR import SparseGPREnsemble

from gpaw import FermiDirac, Mixer

# Costum-made
# from scripts.amorph_struct_randomize import AmorphStructRandomize
# from scripts.amorph_permutation import AmorphPermutationGenerator
# from scripts.add_B_concentration import add_B_concentration

from fcc_HEA_generator import HighEntropyGenerator#I changed v2
from HEA_permutation_generator import PermutationGenerator

import json

# =====================
# Load input JSON
# =====================

json_path = "output_with_base.json"

with open(json_path) as f:
    json_data = json.load(f)
    

# =====================
# Main loop over HEA structures
# =====================

for hea_idx in range(len(json_data["structures"])):

    if hea_idx > 2: # limit
        break
    
    print("=" * 80)
    print(f"HEA INDEX = {hea_idx}")
    print("=" * 80)
    
    structure = json_data["structures"][hea_idx]
    formula = structure["formula"] 
    elements = structure["elements"]
    base = structure["valid_bases"][0]["base"]
    
    # =====================
    # Run settings
    # =====================
            
    seed_HEA = np.random.SeedSequence().entropy
    seed_agox = 0

    sample_size = 10
    num_candidates = {0: [5, 5], 26: [0, 10]}
    
    N_iterations = 100 # AGOX iterations
    ncores = 1 # 24 # 16 genkai
    kpts = (2, 2, 2) 
    
    # =====================
    # Output directories
    # =====================

    path_result = f"0_result"
    path_xsf = f"{path_result}/0_xsf"
    path_fig = f"{path_result}/1_fig"
    db_dir = f"{path_result}/2_db"

    for d in [path_xsf, path_fig, db_dir]:
        os.makedirs(d, exist_ok=True)
    
    # =====================
    # Structure construction
    # =====================
    
    a = base
    
    cell = [
        [2*a, 0,   0],
        [0,   2*a, 0],
        [0,   0,   2*a]
    ]

    template = Atoms("", cell=cell, pbc=True)

    confinement_cell = template.cell.copy()
    confinement_corner = np.array([0, 0, 0])

    # FCC 2x2x2 fractional sites (32 atoms)
    frac_positions = [
        (0,   0,   0),#←第1層兼第5層スタート
        (1/2, 0,   0),
        (0,   1/2, 0),
        (1/4, 1/4, 0),
        (3/4, 1/4, 0),
        (1/4, 3/4, 0),
        (3/4, 3/4, 0),
        (1/2, 1/2, 0),
        
        (1/4, 0, 1/4),#←第2層スタート
        (3/4, 0, 1/4),
        (0, 1/4, 1/4),
        (1/2, 1/4, 1/4),
        (1/4, 1/2, 1/4),
        (3/4, 1/2, 1/4),
        (0, 3/4, 1/4),
        (1/2, 3/4, 1/4),
        
        (0, 0, 1/2),#←第3層スタート
        (1/2, 0, 1/2),
        (0, 1/2, 1/2),
        (1/2, 1/2, 1/2),
        (1/4, 1/4, 1/2),
        (3/4, 1/4, 1/2),
        (1/4, 3/4, 1/2),
        (3/4, 3/4, 1/2),

        (1/4, 0, 3/4),#←第4層スタート
        (3/4, 0, 3/4),
        (0, 1/4, 3/4),
        (1/2, 1/4, 3/4),
        (1/4, 1/2, 3/4),
        (3/4, 1/2, 3/4),
        (0,  3/4, 3/4),
        (1/2, 3/4, 3/4),
        
    ]
    
    environment = Environment(
        template=template,
        symbols=formula,
        confinement_cell=confinement_cell,
        confinement_corner=confinement_corner,
        box_constraint_pbc=[True, True, True],  # Confinement is periodic in all directions.
    )

    database = Database(filename=f"{db_dir}/db_{hea_idx}_{formula}.db", order=5)

    # =====================
    # Generators
    # =====================
    
    generators = [
        HighEntropyGenerator(
            **environment.get_confinement(),
            a=a,
            cell=cell,
            frac_positions=frac_positions,
            use_initialMagmom=True,
            elements=elements,
            json_path=None,
            ),
        
        PermutationGenerator(
            **environment.get_confinement(),
            max_number_of_swaps=5,
            rattle_strength=0.0
            ),
    ]

    # quick sanity check (single generation)
    hea_candidate = generators[0](sampler=None, environment=environment)[0]
    write(f"{path_xsf}/hea_candidate_{hea_idx}_{formula}.xsf", hea_candidate)

    sampler_test = FixedSampler(hea_candidate)
    write(f"{path_xsf}/hea_permut_candidate_{hea_idx}_{formula}.xsf", generators[1](sampler_test, environment)[0])

    # multi sanity check (multi generation)
    # for i in range(100):
    #     hea_candidate = generators[0](sampler=None, environment=environment)[0]
    #     write(f"{path_xsf}/hea_candidate_{i}.xsf", hea_candidate)
    #     print(f"amorph {i}")
        
    #     sampler = FixedSampler(hea_candidate)
    #     write(f"{path_xsf}/hea_permut_candidate_{i}.xsf", generators[1](sampler, environment)[0])
    #     print(f"permut {i}")

    # =====================
    # ML descriptor + model
    # =====================
    
    descriptor = SOAP(environment=environment, weight=True, periodic=True, crossover=True)
    beta = 0.01
    k0 = C(beta, (beta, beta)) * RBF()
    k1 = C(1 - beta, (1 - beta, 1 - beta)) * RBF()
    kernel = C(5000, (1, 1e5)) * (k0 + k1) + Noise(0.01, (0.01, 0.01))
    model = SparseGPR(descriptor=descriptor, kernel=kernel, database=database, prior=Repulsive())
    
    # ---- Model options (keep alternatives) ----
    """
    # model = GPR(descriptor=descriptor, kernel=kernel, database=database, prior=Repulsive())
    # model = SparseGPREnsemble(descriptor=descriptor, kernel=kernel, database=database, prior=Repulsive(), N_ensemble=6) # N_ensem. gpr num
    """
    
    # =====================
    # Sampling + acquisition
    # =====================
    
    sampler = KMeansSampler(descriptor=descriptor, database=database, sample_size=sample_size)

    collector = ParallelCollector(
        generators=generators,
        sampler=sampler,
        environment=environment,
        num_candidates=num_candidates,
        order=1,
    )

    acquisitor = LowerConfidenceBoundAcquisitor(model=model, kappa=2, order=3)

    relaxer = ParallelRelaxPostprocess(
        model=acquisitor.get_acquisition_calculator(),
        constraints=environment.get_constraints(),
        optimizer_run_kwargs={"steps": 1},
        start_relax=8,
        order=2,
    )
    
    # =====================
    # DFT calculator (GPAW)
    # =====================

    calc = SubprocessGPAW(
        ncores=ncores,
        mode={"name": "lcao"},
        basis="dzp",
        xc="PBE",  #exchange correlation
        kpts=kpts,
        spinpol=True,
        symmetry='off',
        occupations={"name": "fermi-dirac", "width": 0.05},
        convergence={'energy': 1e-4, "density": 1e-3, "eigenstates": 1e-3},
        txt='scf.out',
        mixer={"backend": "pulay", "beta": 0.05, "nmaxold": 5, "weight": 100},
        hund=True,
        nbands='nao',
        maxiter=100,
    )

    evaluator = LocalOptimizationEvaluator(
        calc,
        gets={"get_key": "prioritized_candidates"},
        optimizer_kwargs={"logfile": None},
        optimizer_run_kwargs={"fmax": 0.05, "steps": 1},
        constraints=environment.get_constraints(),
        store_trajectory=False,
        order=4,
    )

    # =====================
    # Optional collector (alternative mode)
    # =====================

    """
    collector = StandardCollector(
        generators=generators,
        sampler=sampler,
        environment=environment,
        num_candidates=num_candidates,
        order=1,
    )
    """
    
    # =====================
    # Run AGOX loop
    # =====================

    agox = AGOX(
        collector,
        relaxer,
        acquisitor,
        evaluator,
        database,
        seed=seed_agox,
    )

    agox.run(N_iterations=N_iterations)