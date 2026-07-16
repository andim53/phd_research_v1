from typing import Optional, List, Dict
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ase.io import write
from agox.databases import Database
from scripts.calculate_relative_energy import calculate_relative_energy
from matplotlib.ticker import AutoMinorLocator

def process_database(
    # --- Input & Output Paths ---
    dir_path: str,
    file_idx: int,
    dir_out: str,
    dir_xsf_traj: str,
    dir_xsf: str,

    # --- Trajectory Export Options ---
    save_individual_trajectories: bool = True, 
    individual_seeds_dir_name = "seeds", 

    # --- Iteration Filtering ---
    start_iter: Optional[int] = None,  # Lower bound for iteration filtering
    end_iter: Optional[int] = None,    # Upper bound for iteration filtering

    # --- Plotting Configuration ---
    plot_best_so_far: bool = True,
    plot_by_seed: bool = True,
    figsize = (6, 3.5),

    # --- Plot Limits ---
    custom_max_x: Optional[float] = None,  
    custom_max_y: Optional[float] = None,
) -> None:
    """
    Processes AGOX database files to extract structural data, filter iterations, 
    and output consolidated trajectories, data tables, and progression plots.

    Parameters:
    -----------
    dir_path : str
        Path to the root directory containing the database or seed folders.
    file_idx : int
        Index label used for saving the output files uniquely.
    dir_out : str
        Base output directory path.
    dir_xsf_traj : str
        Relative subdirectory name for storing trajectory files.
    dir_xsf : str
        Relative subdirectory name for storing individual XSF structures.
    plot_best_so_far : bool, optional
        Whether to generate and save a best-so-far optimization progression plot.
    plot_by_seed : bool, optional
        Whether to plot individual traces for each seed.
    save_individual_trajectories : bool, optional
        Whether to save separated structural trajectory files for each individual seed.
    individual_seeds_dir_name : str, optional
        Custom directory name to write the individual seed trajectories into (defaults to "seeds").
    figsize : tuple, optional
        Figure size for the generated progression plot.
    custom_max_x : float, optional
        Custom upper limit for the plot's X-axis (Evaluated Candidates).
    custom_max_y : float, optional
        Custom upper limit for the plot's Y-axis (Energy differential).
    start_iter : int, optional
        The starting iteration number to begin filtering structure data (inclusive).
    end_iter : int, optional
        The ending iteration number to stop filtering structure data (inclusive).
    """
    root = Path(dir_path)
    
    # Locate and sort seed directories naturally based on their trailing integers (e.g., seed_0, seed_1, ...)
    seed_dirs = sorted(
        root.glob("seed_*"), 
        key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else x.name
    )
    
    seed_data: Dict[str, List] = {}

    def load_and_filter_db(db_path: Path) -> List:
        """Helper function to load data from an AGOX SQLite DB, filter by iterations, and return ASE Atoms."""
        db = Database(filename=str(db_path))
        raw_structures = db.get_all_structures_data()
        
        # Apply strict inclusive bounds filtering on the iteration count
        if start_iter is not None:
            raw_structures = [d for d in raw_structures if d.get("iteration", 0) >= start_iter]
        if end_iter is not None:
            raw_structures = [d for d in raw_structures if d.get("iteration", 0) <= end_iter]
            
        # Convert raw database dictionaries into operational ASE Atoms objects
        return [db.db_to_atoms(struct) for struct in raw_structures]

    # Handle flat structure execution (no seed directories present) vs multi-seed structures
    if not seed_dirs:
        db_path = root / "1_db" / "db_0.db"
        if db_path.exists():
            seed_data["Seed 0"] = load_and_filter_db(db_path)
    else:
        # Standardize seed labels sequentially starting from 0, matching chronological execution order
        for i, p in enumerate(seed_dirs):
            seed_num = p.name.split('_')[-1]
            db_path = p / "1_db" / f"db_{seed_num}.db"
            if db_path.exists():
                seed_data[f"Seed {i}"] = load_and_filter_db(db_path)

    # Flatten nested dictionary trajectories into a unified global timeline list
    all_traj = []
    sorted_seed_names = sorted(seed_data.keys(), key=lambda x: int(x.split()[-1]))
    for s_name in sorted_seed_names:
        all_traj.extend(seed_data[s_name])

    # Early exit if the specified iteration window yields no structures
    if not all_traj:
        print(f"No structures found matching the iteration scope [{start_iter} to {end_iter}].")
        return

    # Setup file output path trees
    out_base = Path(dir_out)
    traj_dir = out_base / dir_xsf_traj
    xsf_dir = out_base / dir_xsf / str(file_idx)
    traj_dir.mkdir(parents=True, exist_ok=True)
    xsf_dir.mkdir(parents=True, exist_ok=True)

    # Export unified master trajectories (both in XSF and native ASE Trajectory formats)
    write(traj_dir / f"traj_{file_idx}.xsf", all_traj)
    write(traj_dir / f"traj_{file_idx}.traj", all_traj)

    # Optionally isolate and save structural files for individual seed histories
    if save_individual_trajectories:
        # Evaluates the 'individual_seeds_dir_name' argument to isolate separate seed paths
        seed_traj_dir = traj_dir / str(individual_seeds_dir_name)
        seed_traj_dir.mkdir(parents=True, exist_ok=True)
        for s_name, s_traj in seed_data.items():
            if not s_traj:  
                continue
            clean_name = s_name.lower().replace(" ", "_")
            write(seed_traj_dir / f"{file_idx}_{clean_name}.traj", s_traj)
            write(seed_traj_dir / f"{file_idx}_{clean_name}.xsf", s_traj)
    
    # Calculate energy differentials using internal helper script
    energies, _, rel_e, rel_e_atom = calculate_relative_energy(all_traj)

    # Write out data values into a standardized DataFrame table
    df = pd.DataFrame({
        "index": range(len(all_traj)),
        "energy": energies,
        "relative_energy": rel_e,
        "relative_energy_per_atom": rel_e_atom
    })
    df.to_csv(out_base / f"data_{file_idx}.csv", index=False)

    # Strip magnetic moment arrays to avoid visualization compatibility errors in individual XSF steps
    for i, frame in enumerate(all_traj):
        frame_i = frame.copy()
        for key in ["initial_magmoms", "magmoms"]:
            frame_i.arrays.pop(key, None)
        write(xsf_dir / f"struct_{i}.xsf", frame_i)

    # --- Best-So-Far Progression Plotting ---
    if plot_best_so_far:
        fig, ax = plt.subplots(figsize=figsize)
        max_y = 0
        max_x = 0

        if plot_by_seed:
            cmap = plt.get_cmap('tab10')
            for i, s_name in enumerate(sorted_seed_names):
                s_traj = seed_data[s_name]
                if not s_traj:
                    continue
                _, _, _, s_rel_e_atom = calculate_relative_energy(s_traj)
                
                # Compute progressive minimal convergence threshold over time
                s_best_so_far = np.minimum.accumulate(s_rel_e_atom)
                
                # Visual hierarchy tuning: Highlight Seed 0 in bold black on top
                if i == 0:
                    current_color = 'black'
                    linewidth = 2.0  
                    zorder = 50      
                else:
                    current_color = cmap((i-1) % 10)  # Rotate remaining qualitative color map steps
                    linewidth = 1.5
                    zorder = 1

                ax.plot(range(len(s_best_so_far)), s_best_so_far, 
                        label=s_name, lw=linewidth, color=current_color, zorder=zorder)
                
                max_y = max(max_y, np.max(s_rel_e_atom))
                max_x = max(max_x, len(s_best_so_far))
        else:
            # Flattened singular optimization history trace
            best_so_far = np.minimum.accumulate(rel_e_atom)
            ax.plot(range(len(best_so_far)), best_so_far, color='black', lw=2, label='Combined')
            max_y = np.max(rel_e_atom)
            max_x = len(best_so_far)

        # Plot formatting & styling adjustments
        ax.set_xlabel('Evaluated Candidates')
        ax.set_ylabel(r'$E_{i}-E_{glob}$ (eV/atom)')

        final_xlim = custom_max_x if custom_max_x is not None else max_x
        final_ylim = custom_max_y if custom_max_y is not None else max_y * 1.1

        ax.set_xlim(0, final_xlim)
        ax.set_ylim(0, final_ylim)
        
        # Add high-density structural sub-grid intervals 
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Clean up chartjunk by removing top and right boundaries
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(
            axis='both',          
            which='both',         
            top=False,            
            right=False,          
            labeltop=False,       
            labelright=False      
        )

        # Draw the descriptive labels clear of the actual plotting curves outside of the chart area
        if plot_by_seed:
            ax.legend(
                loc='upper left', 
                fontsize=9, 
                ncol=1, 
                frameon=True,
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0.,
            )
            
        plt.tight_layout()
        
        # Save structural image output
        plot_path = out_base / "progression_plots"
        plot_path.mkdir(exist_ok=True)
        plt.savefig(plot_path / f"progression_seed_split_{file_idx}.png", dpi=300)
        plt.show()
        plt.close(fig)