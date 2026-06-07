import os
import numpy as np

from ase import Atoms
from ase.build import bulk
from ase.io import write, read

# package untuk ML
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

# costume generator; bragg generator
from scripts.bragg_generator import BraggGenerator

# addition untuk buat structure
from ase.build import bulk
from ase.build import make_supercell

############################ parameter
cif_file = "FePt.cif" #"Fe3Pt.cif"
S_target = 0.9

# Opsi A: Tentukan nilai konstanta kisi baru [a, b, c, alpha, beta, gamma]
# Contoh: a=3.9, b=3.9, c=3.7, alpha=90, beta=90, gamma=90
new_lattice_params = [3.90, 3.90, 3.70, 90, 90, 90]

supercell = (3, 3, 3)

kpts = (1, 1, 1) 
ncores = 24 # 16 genkai

# param Agox
N_iterations = 100 # AGOX iterations
num_candidates = {0: [10, 0], 10: [5, 5]} # control generator
sample_size = 10

############################

# input structure reference
atoms = read(cif_file)

# edit lattice parameter
atoms.set_cell(new_lattice_params, scale_atoms=True)

# edit supercell
P = np.diag(supercell)
atoms = make_supercell(atoms, P)

success_count = 0
fail_count = 0

for seed in range(100):
    print(f"Start seed: {seed}")   

    path_result = f"seed_{seed}/0_result"
    path_xsf = f"{path_result}/0_xsf"
    path_fig = f"{path_result}/1_fig"
    db_dir = f"seed_{seed}/1_db"

    for d in [path_xsf, path_fig, db_dir]:
        os.makedirs(d, exist_ok=True)

    # environment
    template = Atoms("", cell=atoms.cell.copy(), pbc=True)
    environment = Environment(
        template=template,
        symbols=atoms.get_chemical_formula(),
        confinement_cell=template.cell.copy(),
        confinement_corner=np.array([0, 0, 0]),
        box_constraint_pbc=[True, True, True],
    )    

    n_rattle = len(atoms)
    generators = [
        BraggGenerator(
        **environment.get_confinement(),
        atoms_template = atoms,
        S_target = S_target, # 1.0,
        # S_options=None,
        # seed=2,
        randomize_seed=True,
        write_struct=True,
        output_dir="generated_structures",
        min_distance_scale=0.7,
        print_result=False,
        replace=True
        ),
        
        RattleGenerator(
            **environment.get_confinement(),
            n_rattle=int(n_rattle), # 0.7 * n_rattle
            rattle_amplitude=0.3 #2.3,
        ),
        # PermutationGenerator(
        #     **environment.get_confinement(),
        #     max_number_of_swaps=n_atoms,
        #     rattle_strength=0.3,
        #     use_xy_only=False,
        #     ignore_H=False,
        #     write_candidates_to_disk=False,
        #     replace=True,
        # )
    ]

    # test generate structure
    try:
            bragg_candidate = generators[0](
                sampler=None,
                environment=environment
            )[0]
            write(f"{path_xsf}/bragg_candidate_{seed}.xsf", bragg_candidate)
            print(f"bragg {seed}")
        
            sampler_test = FixedSampler(bragg_candidate)
        
            rattle_candidate = generators[1](
                sampler_test,
                environment
            )[0]
            write(f"{path_xsf}/rattle_candidate_{seed}.xsf", rattle_candidate)
            print(f"rattle {seed}")
        
            # permut_candidate = generators[2](
            #     sampler_test,
            #     environment
            # )[0]
            # write(f"{path_xsf}/permut_candidate_{seed}.xsf", permut_candidate)
            # print(f"permut {seed}")
            success_count += 1
    except Exception:
            fail_count += 1

    print(f"Current: {success_count=}, {fail_count=}")

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
    