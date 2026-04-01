from typing import Optional
from pathlib import Path
import pandas as pd
from ase.io import write
from agox.databases import Database
from .calculate_relative_energy import calculate_relative_energy

def process_database(
    dir_path: str,
    file_idx: int,
    dir_out: str,
    dir_xsf_traj: str,
    dir_xsf: str,
) -> None:
    """
    Extracts trajectory data from AGOX databases, calculates relative energies, 
    and saves structures and CSV data to specified output directories.
    """
    root = Path(dir_path)
    seeds = sorted(root.glob("seed_*"))
    traj = []

    if not seeds:
        db_path = root / "1_db" / "db_0.db"
        if db_path.exists():
            db = Database(filename=str(db_path))
            db.restore_to_memory()
            traj = db.restore_to_trajectory()
    else:
        for p in seeds:
            seed_num = p.name.split('_')[-1]
            db_path = p / "1_db" / f"db_{seed_num}.db"
            if db_path.exists():
                db = Database(filename=str(db_path))
                db.restore_to_memory()
                traj.extend(db.restore_to_trajectory())

    if not traj:
        return

    traj_dir = Path(dir_out) / dir_xsf_traj
    xsf_dir = Path(dir_out) / dir_xsf / str(file_idx)
    
    traj_dir.mkdir(parents=True, exist_ok=True)
    xsf_dir.mkdir(parents=True, exist_ok=True)

    write(traj_dir / f"traj_{file_idx}.xsf", traj)
    write(traj_dir / f"traj_{file_idx}.traj", traj)

    energies, _, rel_e, rel_e_atom = calculate_relative_energy(traj)

    df = pd.DataFrame({
        "index": range(len(traj)),
        "energy": energies,
        "relative_energy": rel_e,
        "relative_energy_per_atom": rel_e_atom
    })
    df.to_csv(Path(dir_out) / f"data_{file_idx}.csv", index=False)

    for i, frame in enumerate(traj):
        frame_i = frame.copy()
        for key in ["initial_magmoms", "magmoms"]:
            frame_i.arrays.pop(key, None)
        write(xsf_dir / f"struct_{i}.xsf", frame_i)