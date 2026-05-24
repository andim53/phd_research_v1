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
from agox.models.GPR import GPR
from agox.models.GPR.kernels import RBF, Noise, Constant as C
from agox.models.GPR.priors import Repulsive
from agox.helpers import SubprocessGPAW

# Costum-made
from scripts.amorph_struct_randomize import AmorphStructRandomize
from scripts.amorph_permutation import AmorphPermutationGenerator
from scripts.add_B_concentration import add_B_concentration

"""
Cell volume controls

Helps for permutation and random movement.
"""
scale_cell = 1.01 # 1.05->+5%

a_ta = 3.30
supercell = (3, 3, 3)
kpts = (1, 1, 1) 
ncores = 24 # 16 genkai
cB = 0.01 # B concentrations (%)
N_iterations = 100 # AGOX iterations
num_candidates = {0: [10, 0, 0], 10: [5, 5, 0], 25: [2, 3, 5]}
sample_size = 10

for seed in range(100):
    print(f"Start seed: {seed}")   

    path_result = f"seed_{seed}/0_result"
    path_xsf = f"{path_result}/0_xsf"
    path_fig = f"{path_result}/1_fig"
    db_dir = f"seed_{seed}/1_db"

    for d in [path_xsf, path_fig, db_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Make ta with Boron
    ta_bulk = bulk("Ta", "bcc", a=a_ta, cubic=True) * supercell
    ta_bulk.set_cell(ta_bulk.cell * scale_cell, scale_atoms=True)

    tab_bulk, n_b, n_ta = add_B_concentration(
        ta_bulk, cB=cB, make_structure=True
    )

    """
    Environment is an empty cells, size of ta_bulk. 
    """
    template = Atoms("", cell=ta_bulk.cell.copy(), pbc=True)
    environment = Environment(
        template=template,
        symbols=tab_bulk.get_chemical_formula(),
        confinement_cell=template.cell.copy(),
        confinement_corner=np.array([0, 0, 0]),
        box_constraint_pbc=[True, True, True],
    )

    n_rattle = len(ta_bulk)
    generators = [
        AmorphStructRandomize(
            **environment.get_confinement(),
            amorph=tab_bulk,
            write_struct=True,
            rattle_amplitude=1.5,
            n_rattle=n_rattle,
            generate_pristine=False,
        ),
        RattleGenerator(
            **environment.get_confinement(),
            n_rattle=int(n_rattle), # 0.7 * n_rattle
            rattle_amplitude=2.3,
        ),
        AmorphPermutationGenerator(
            **environment.get_confinement(),
            max_number_of_swaps=5,
            rattle_strength=0.0,
            # attempts=1,
            # confinement_check = True,
            # check_overlap=True,		
        ),
    ]

    amorph_candidate = generators[0](sampler=None, environment=environment)[0]
    write(f"{path_xsf}/amorph_candidate.xsf", amorph_candidate)

    sampler = FixedSampler(amorph_candidate)
    write(f"{path_xsf}/rattle_candidate.xsf", generators[1](sampler, environment)[0])
    write(f"{path_xsf}/permut_candidate.xsf", generators[2](sampler, environment)[0])

    """
    for i in range(100):
        amorph_candidate = generators[0](sampler=None, environment=environment)[0]
        write(f"{path_xsf}/amorph_candidate_{i}.xsf", amorph_candidate)
        print(f"amorph {i}")
        
        sampler = FixedSampler(amorph_candidate)
        write(f"{path_xsf}/permut_candidate_{i}.xsf", generators[1](sampler, environment)[0])
        print(f"permut {i}")
        
        write(f"{path_xsf}/rattle_candidate_{i}.xsf", generators[2](sampler, environment)[0])
        print(f"rattle {i}")
    """

    database = Database(filename=f"{db_dir}/db_{seed}.db", order=5)
    descriptor = Fingerprint(environment=environment)  

    beta = 0.01
    k0 = C(beta, (beta, beta)) * RBF()
    k1 = C(1 - beta, (1 - beta, 1 - beta)) * RBF()
    kernel = C(5000, (1, 1e5)) * (k0 + k1) + Noise(0.01, (0.01, 0.01))

    model = GPR(
        descriptor=descriptor,
        kernel=kernel,
        database=database,
        prior=Repulsive(),
        # use_ray = False
    )

    sampler = KMeansSampler(
        descriptor=descriptor,
        database=database,
        sample_size=sample_size,
    )

    collector = ParallelCollector(
        generators=generators,
        sampler=sampler,
        environment=environment,
        num_candidates=num_candidates,
        order=1,
    )

    """
    collector = StandardCollector(
        generators=generators,
        sampler=sampler,
        environment=environment,
        num_candidates=num_candidates,
        order=1,
    )	
    """

    acquisitor = LowerConfidenceBoundAcquisitor(model=model, kappa=2, order=3)

    relaxer = ParallelRelaxPostprocess(
        model=acquisitor.get_acquisition_calculator(),
        constraints=environment.get_constraints(),
        optimizer_run_kwargs={"steps": 100},
        start_relax=10,
        order=2,
    )

    calc = SubprocessGPAW(
        ncores=ncores,
        mode={"name": "lcao"},
        basis="dzp",
        xc="PBE",
        kpts=kpts,
        symmetry="off",
        nbands="nao",
        mixer={"backend": "pulay", "beta": 0.05, "nmaxold": 5, "weight": 100},
        convergence={"energy": 1e-4, "density": 1e-3, "eigenstates": 1e-3},
        occupations={"name": "fermi-dirac", "width": 0.05},
        maxiter=100,
        txt="output.txt",
    )

    evaluator = LocalOptimizationEvaluator(
        calc,
        gets={"get_key": "prioritized_candidates"},
        optimizer_kwargs={"logfile": None},
        optimizer_run_kwargs={"fmax": 0.05, "steps": 1},
        constraints=environment.get_constraints(),
        order=4,
    )

    agox = AGOX(
        collector,
        relaxer,
        acquisitor,
        evaluator,
        database,
        seed=seed,
    )

    agox.run(N_iterations=N_iterations)