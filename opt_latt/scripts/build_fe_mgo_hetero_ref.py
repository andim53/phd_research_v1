from typing import Tuple
from ase import Atoms
from ase.build import surface, bulk
from ase.io import write

from .build_mgo_stack import build_mgo_stack
from .build_fe_stack import build_fe_stack


def build_fe_mgo_hetero_ref(
    dir_out: str,
    dir_xsf: str,
    a_mgo: float = 4.212,
    a_fe: float = 2.870190,
    vacuum: float = 10.0,
    dist_z_fe2o: float = 2.3,
    num_layers_mgo: int = 1,
    num_layers_fe: int = 1,
    super_cell: Tuple[int, int, int] = (3, 3, 1),
    save_path: str = f"./struct_ref.xsf"
) -> Atoms:
    """
    Build Fe/MgO heterostructure and save an .xsf reference structure.

    Args:
        dir_out: Path where output directory resides.
        dir_xsf: Folder name where file is written.
        a_mgo: MgO lattice constant.
        a_fe: Fe lattice constant (not explicitly built here, but kept for clarity).
        vacuum: Vacuum spacing added along z.
        dist_z_fe2o: Vertical spacing between Fe and MgO blocks.
        num_layers_mgo: Number of MgO layers to build.
        num_layers_fe: Number of Fe layers to build.
        super_cell: (nx, ny, nz) repetition.

    Returns:
        Final heterostructure `Atoms` object.
    """

    # MgO primitive slab and spacing
    bulk_mgo = bulk("MgO", "rocksalt", a=a_mgo, cubic=True)
    slab_mgo_prim = surface(bulk_mgo, (0, 0, 1), layers=1, vacuum=vacuum)
    dist_z_mgo = slab_mgo_prim[5].position[2] - slab_mgo_prim[0].position[2]

    # Fe primitive slab
    slab_fe_prim = surface("Fe", (0, 0, 1), layers=1, vacuum=vacuum)

    # MgO grown above Fe gives correct interface separation; then remove Fe
    slab_mgofe = build_mgo_stack(
        slab_fe_prim,
        num_layers=num_layers_mgo,
        dist_fe2o=dist_z_fe2o,
        dist_mgo=dist_z_mgo,
    )
    slab_mgo = slab_mgofe[[atom.symbol != "Fe" for atom in slab_mgofe]]

    # Fe multilayer block
    slab_fe = build_fe_stack(
        slab_fe_prim,
        num_layers=num_layers_fe,
        vacuum=vacuum,
    )

    # Replicate laterally
    slab_mgo = slab_mgo.repeat(super_cell)
    slab_fe = slab_fe.repeat(super_cell)

    # Copy for alignment
    slab_substrate = slab_mgo.copy()
    slab_deposition = slab_fe.copy()

    # Vertical alignment: place Fe above MgO
    z_sub_max = slab_substrate.positions[:, 2].max()
    z_dep_min = slab_deposition.positions[:, 2].min()
    slab_deposition.translate((0, 0, z_sub_max - z_dep_min + dist_z_fe2o))

    # Build final structure
    slab_hetero = slab_substrate + slab_deposition

    write(save_path, slab_hetero)
    return slab_hetero

# build_fe_mgo_hetero_ref(
#     dir_out = dir_out,
#     dir_xsf = dir_xsf,
#     a_mgo= 4.212,
#     a_fe= 2.870190,
#     vacuum= 10.0,
#     dist_z_fe2o= 2.3,
#     num_layers_mgo= 1,
#     num_layers_fe= 1,
#     super_cell= (3, 3, 1)
# )