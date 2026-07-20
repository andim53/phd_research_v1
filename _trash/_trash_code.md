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
    dir_path: str,
    file_idx: int,
    dir_out: str,
    dir_xsf_traj: str,
    dir_xsf: str,
    plot_best_so_far: bool = True,
    plot_by_seed: bool = True,
    save_individual_trajectories: bool = True, 
    individual_seeds_dir_name = "seeds",  # New parameter to customize the seed output subdirectory name
    figsize = (6, 3.5),
    custom_max_x: Optional[float] = None,  
    custom_max_y: Optional[float] = None,
    start_iter: Optional[int] = None,  
    end_iter: Optional[int] = None,    
) -> None:
    """
    Processes AGOX database files to extract structural data, filter iterations, 
    and output consolidated trajectories, data tables, and progression plots.

    - Dynamically detects and sequentially renames seed directories to start at 0.
    - Filters structures by a user-defined iteration window [start_iter, end_iter].
    - Computes and saves relative energy metrics per atom.
    - Generates standardized progression plots showcasing optimization performance.
    """
    root = Path(dir_path)
    
    # Locate and sort seed directories naturally based on their trailing integers
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
        # Standardize seed labels sequentially from 0 matching chronological execution order
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
        # Dynamically use the custom directory name parameter here
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
    dir_path: str,
    file_idx: int,
    dir_out: str,
    dir_xsf_traj: str,
    dir_xsf: str,
    plot_best_so_far: bool = True,
    plot_by_seed: bool = True,
    save_individual_trajectories: bool = True, 
    figsize = (6, 3.5),
    custom_max_x: Optional[float] = None,  
    custom_max_y: Optional[float] = None,
    start_iter: Optional[int] = None,  # New parameter
    end_iter: Optional[int] = None,    # New parameter
) -> None:
    """
    Extracts trajectory data with custom iteration-range filtering. 
    Seeds are renamed to start at 0 sequentially for cleaner labeling.
    """
    root = Path(dir_path)
    # Sort seeds naturally by name
    seed_dirs = sorted(root.glob("seed_*"), key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else x.name)
    
    seed_data: Dict[str, List] = {}

    def load_and_filter_db(db_path: Path) -> List:
        """Helper to get data raw, apply filters, and convert to ASE Atoms."""
        db = Database(filename=str(db_path))
        raw_structures = db.get_all_structures_data()
        
        # Apply iteration filters if provided
        if start_iter is not None:
            raw_structures = [d for d in raw_structures if d.get("iteration", 0) >= start_iter]
        if end_iter is not None:
            raw_structures = [d for d in raw_structures if d.get("iteration", 0) <= end_iter]
            
        # Convert the filtered raw dictionary data into ASE Atoms objects
        return [db.db_to_atoms(struct) for struct in raw_structures]

    if not seed_dirs:
        db_path = root / "1_db" / "db_0.db"
        if db_path.exists():
            seed_data["Seed 0"] = load_and_filter_db(db_path)
    else:
        # Renaming logic: Always start from 0 based on the order found
        for i, p in enumerate(seed_dirs):
            seed_num = p.name.split('_')[-1]
            db_path = p / "1_db" / f"db_{seed_num}.db"
            if db_path.exists():
                # Create a standardized name: "Seed 0", "Seed 1", etc.
                seed_data[f"Seed {i}"] = load_and_filter_db(db_path)

    # Flatten for combined files
    all_traj = []
    # Using sorted keys ensures we process Seed 0, Seed 1, etc., in order
    sorted_seed_names = sorted(seed_data.keys(), key=lambda x: int(x.split()[-1]))
    for s_name in sorted_seed_names:
        all_traj.extend(seed_data[s_name])

    if not all_traj:
        print(f"No structures found matching the iteration scope [{start_iter} to {end_iter}].")
        return

    # Setup directories
    out_base = Path(dir_out)
    traj_dir = out_base / dir_xsf_traj
    xsf_dir = out_base / dir_xsf / str(file_idx)
    traj_dir.mkdir(parents=True, exist_ok=True)
    xsf_dir.mkdir(parents=True, exist_ok=True)

    # Save trajectories and CSV
    write(traj_dir / f"traj_{file_idx}.xsf", all_traj)
    write(traj_dir / f"traj_{file_idx}.traj", all_traj)

    if save_individual_trajectories:
        seed_traj_dir = traj_dir / "seeds"
        seed_traj_dir.mkdir(exist_ok=True)
        for s_name, s_traj in seed_data.items():
            if not s_traj:  # Skip empty seeds after filtration
                continue
            clean_name = s_name.lower().replace(" ", "_")
            write(seed_traj_dir / f"{file_idx}_{clean_name}.traj", s_traj)
            write(seed_traj_dir / f"{file_idx}_{clean_name}.xsf", s_traj)
    
    energies, _, rel_e, rel_e_atom = calculate_relative_energy(all_traj)

    df = pd.DataFrame({
        "index": range(len(all_traj)),
        "energy": energies,
        "relative_energy": rel_e,
        "relative_energy_per_atom": rel_e_atom
    })
    df.to_csv(out_base / f"data_{file_idx}.csv", index=False)

    # Individual XSF saving
    for i, frame in enumerate(all_traj):
        frame_i = frame.copy()
        for key in ["initial_magmoms", "magmoms"]:
            frame_i.arrays.pop(key, None)
        write(xsf_dir / f"struct_{i}.xsf", frame_i)

    # --- Best-So-Far Plotting ---
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
                s_best_so_far = np.minimum.accumulate(s_rel_e_atom)
                
                # --- Specific Color Logic ---
                if i == 0:
                    current_color = 'black'
                    linewidth = 2.0  # Slightly thicker to emphasize Seed 0
                    zorder = 50      # Ensures Seed 0 is drawn on top of others
                else:
                    # Offset the color index so we don't waste the first cmap color
                    current_color = cmap((i-1) % 10) 
                    linewidth = 1.5
                    zorder = 1

                ax.plot(range(len(s_best_so_far)), s_best_so_far, 
                        label=s_name, lw=linewidth, color=current_color, zorder=zorder)
                
                max_y = max(max_y, np.max(s_rel_e_atom))
                max_x = max(max_x, len(s_best_so_far))
        else:
            best_so_far = np.minimum.accumulate(rel_e_atom)
            ax.plot(range(len(best_so_far)), best_so_far, color='black', lw=2, label='Combined')
            max_y = np.max(rel_e_atom)
            max_x = len(best_so_far)

        # Formatting
        ax.set_xlabel('Evaluated Candidates')
        ax.set_ylabel(r'$E_{i}-E_{glob}$ (eV/atom)')

        final_xlim = custom_max_x if custom_max_x is not None else max_x
        final_ylim = custom_max_y if custom_max_y is not None else max_y * 1.1

        ax.set_xlim(0, final_xlim)
        ax.set_ylim(0, final_ylim)
        
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(
            axis='both',          # Apply to both x and y axes
            which='both',         # Apply to both major and minor ticks
            top=False,            # Turn off ticks on the top
            right=False,          # Turn off ticks on the right
            labeltop=False,       # Turn off labels on the top
            labelright=False      # Turn off labels on the right
        )

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
        
        plot_path = out_base / "progression_plots"
        plot_path.mkdir(exist_ok=True)
        plt.savefig(plot_path / f"progression_seed_split_{file_idx}.png", dpi=300)
        plt.show()
        plt.close(fig)

import numpy as np
from scipy.stats import gaussian_kde
from ase.io import read

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

dir_path = f"{dir_out}/{dir_xsf_traj}"
traj_files = ["traj_22.traj", "traj_23.traj", "traj_24.traj", "traj_25.traj", "traj_26.traj"]
# traj_files = ["traj_27.traj", "traj_28.traj", "traj_29.traj", "traj_30.traj", "traj_31.traj"] 
# legend_labels = ["0% MgO latt.", "25% MgO latt.", "50% MgO latt.", "75% MgO latt."]
legend_labels = labels

colors = plt.cm.viridis(np.linspace(0, 1, len(traj_files)))

min_e, max_e = 0.0, 0.8
energy_grid = np.linspace(min_e, max_e, 200)
fig, ax = plt.subplots(figsize=(3, 4))

for i, filename in enumerate(traj_files):
    try:
        full_path = f"{dir_path}/{filename}"
        structures = read(full_path, index=':')

        raw_energies = []
        for atoms in structures:
            try:
                raw_energies.append(atoms.get_potential_energy())
            except:
                continue
        
        if not raw_energies:
            continue
            
        num_atoms = len(structures[0])
        energies = (np.array(raw_energies) - min(raw_energies)) / num_atoms

        kde = gaussian_kde(energies)
        density = kde.evaluate(energy_grid)
        label = filename.replace('.traj', '')

        ax.plot(density, energy_grid, color=colors[i], lw=1, label=legend_labels[i], zorder=4)
        ax.fill_betweenx(energy_grid, 0, density, color=colors[i], alpha=0.1)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

ax.set_ylim(min_e, max_e)
ax.set_xlim(0, None)  # Starts density at 0

ax.set_ylabel(r'$E_{i}-E_{min}$ (eV/atom)', fontsize=12)
ax.set_xlabel('State Density', fontsize=12)

ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(labelsize=11)

ax.legend(frameon=False, loc='upper right')
# ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patheffects as patheffects
from matplotlib.animation import FuncAnimation  # Added for animation
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from typing import Optional, List

def plot_structure_landscape(
    X_eigen, 
    energies,                    # Can be an array, list of arrays, or dict of arrays
    z_data=None, 
    show_colorbar=True, 
    cmap='viridis',
    z_limit=(None, None, 5),
    e_limit=(0.0, 0.8, 5),
    figsize=(6, 3),
    show_limits=True,            # Controls automatic peak rendering
    custom_peak_labels: Optional[List[str]] = None,  # e.g. ["Island", "Flat"]
    fontsize=10, 
    wspace=0.05, 
    cbar_pad=0.02, 
    density_left=True,
    plot_density_only=False,  
    custom_legends=None,         # Explicit text labels list if energies is a list
    density_cmap='Blues',        # Colormap gradient name (e.g., 'Blues', 'Purples', 'magma')
    black_seed_zero=False,       # Force index 0 to be black, remainder to be gradient
    seed_zero_top_zorder=False,  # Force seed 0 to have the highest zorder layer visibility
    fill_density=False,          # Toggle to fill under the curve or not
    density_alpha=0.12,          # Control fill opacity
    dens_line_weight=1.5,
    save_path='./',
    animate_scatter=False,       # NEW: Toggle to enable scatter animation
    animation_fps=20,            # NEW: Control the speed of the animation
    gif_name='conf_space_ani.gif',# NEW: Name of the output animation file
    plot_z_vs_e = False,
):
    min_e, max_e = e_limit[0], e_limit[1]
    eticks = np.round(np.linspace(min_e, max_e, e_limit[2]), 1)
    
    # --- Structure & Normalize Input Energies ---
    energy_datasets = {}
    if isinstance(energies, dict):
        energy_datasets = energies
    elif isinstance(energies, (list, tuple)) and len(energies) > 0 and isinstance(energies[0], (list, np.ndarray)):
        for i, dataset in enumerate(energies):
            label = custom_legends[i] if (custom_legends and i < len(custom_legends)) else f"Dataset {i+1}"
            energy_datasets[label] = np.asarray(dataset)
    else:
        # Fallback for single data array
        label = custom_legends[0] if custom_legends else "State Density"
        energy_datasets[label] = np.asarray(energies)

    # --- Setup Gradient Color Palette ---
    num_curves = len(energy_datasets)
    try:
        color_gradient = plt.colormaps.get_cmap(density_cmap)
    except AttributeError:
        color_gradient = plt.cm.get_cmap(density_cmap)

    sampled_colors = []
    if num_curves > 1:
        if black_seed_zero:
            sampled_colors.append('black')
            remaining_count = num_curves - 1
            if remaining_count > 1:
                color_indices = np.linspace(0.85, 0.35, remaining_count)
                for x in color_indices:
                    sampled_colors.append(color_gradient(x))
            else:
                sampled_colors.append(color_gradient(0.60))
        else:
            color_indices = np.linspace(0.95, 0.35, num_curves)
            sampled_colors = [color_gradient(x) for x in color_indices]
    else:
        sampled_colors = ['black' if black_seed_zero else color_gradient(0.85)]

    # --- Setup Subplots Axis Framework ---
    if plot_density_only:
        width_mod = 0.55 if num_curves > 1 else 0.4
        fig, ax_dens = plt.subplots(figsize=(figsize[0] * width_mod, figsize[1]))
        ax_scat = None
        ax_l, ax_r = ax_dens, None
    else:
        ratios = [1, 2.5] if density_left else [2.5, 1]
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={'width_ratios': ratios})
        fig.subplots_adjust(wspace=wspace)
        ax_scat = ax_r if density_left else ax_l
        ax_dens = ax_l if density_left else ax_r

    # --- Scatter Plot Representation ---
    scat_anim = None
    if not plot_density_only and ax_scat is not None:
        if z_data is not None:
            vmin = z_limit[0] if z_limit[0] is not None else np.min(z_data)
            vmax = z_limit[1] if z_limit[1] is not None else np.max(z_data)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            color_src = np.asarray(z_data)
        else:
            norm = None
            color_src = 'white'

        scat_energies = next(iter(energy_datasets.values())) if num_curves > 1 else list(energy_datasets.values())[0]

        if len(X_eigen) == len(scat_energies):
            # If animating, we start with an empty or single-point scatter representation
            if animate_scatter:
                initial_idx = 1
                initial_colors = color_src[:initial_idx] if z_data is not None else 'white'
                sc = ax_scat.scatter(X_eigen[:initial_idx], scat_energies[:initial_idx], c=initial_colors, s=25,
                                     cmap=cmap if z_data is not None else None,
                                     norm=norm, edgecolors='black', linewidth=0.5,
                                     alpha=0.8, zorder=2)
            else:
                sc = ax_scat.scatter(X_eigen, scat_energies, c=color_src, s=25,
                                     cmap=cmap if z_data is not None else None,
                                     norm=norm, edgecolors='black', linewidth=0.5,
                                     alpha=0.8, zorder=2)
            
            if show_colorbar and z_data is not None:
                cbar = plt.colorbar(sc, ax=ax_scat, pad=cbar_pad)
                cbar.set_label(r'$\Delta z$ (Å)', fontsize=fontsize)
                tick_locs = np.linspace(vmin, vmax, z_limit[2])
                cbar.set_ticks(tick_locs)
                cbar.set_ticklabels([f"{t:.2f}" for t in tick_locs])
        else:
            print("Warning: X_eigen layout dims don't match base layout. Skipping scatter population.")
            animate_scatter = False

    # --- Evaluate & Generate Multiple Gaussian KDE Curves with Gradient Colors ---
    energy_grid = np.linspace(min_e, max_e, 200)
    total_density = np.zeros_like(energy_grid)

    for idx, (name, data_array) in enumerate(energy_datasets.items()):
        if len(data_array) > 1:
            kde = gaussian_kde(data_array)
            density = kde.evaluate(energy_grid)
            total_density += density

            curve_color = sampled_colors[idx]

            if idx == 0 and seed_zero_top_zorder:
                current_line_zorder = 10
                current_fill_zorder = 9
            else:
                current_line_zorder = 4
                current_fill_zorder = 3

            # Plot profiles
            ax_dens.plot(density, energy_grid, color=curve_color, lw=dens_line_weight, zorder=current_line_zorder, label=name)

            if fill_density:
                ax_dens.fill_betweenx(energy_grid, 0, density, color=curve_color, alpha=density_alpha, zorder=current_fill_zorder)

    ax_dens.set_xlabel('State Density (config./eV)', fontsize=fontsize)

    if num_curves > 1:
        fig.legend(
            frameon=False,
            fontsize=fontsize-2,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5)
        )

    if not plot_density_only and ax_scat is not None:
        ax_scat.set_xlabel(r'$\psi_{1d}(a.u.)$', fontsize=fontsize)
        ax_scat.xaxis.set_minor_locator(AutoMinorLocator())
        # Explicitly freeze the limits based on data so axis doesn't dance around during animation
        ax_scat.set_xlim(np.min(X_eigen) - 0.1, np.max(X_eigen) + 0.1)

    if plot_density_only or density_left:
        ax_dens.set_ylabel(r'$E_{i}-E_{glob}$ (eV/atom)', fontsize=fontsize)

    # --- Automatic Peak Detection and Visualization Logic ---
    if show_limits and len(energy_datasets) > 0:
        peaks, _ = find_peaks(total_density, prominence=np.max(total_density)*0.05)
        peaks = sorted(peaks, key=lambda idx: energy_grid[idx])
        axes_to_mark = [ax_dens] if plot_density_only else [ax_l, ax_r]

        for i, peak_idx in enumerate(peaks):
            peak_energy = energy_grid[peak_idx]

            for ax in axes_to_mark:
                if ax is not None:
                    ax.axhline(y=peak_energy, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

            if custom_peak_labels and i < len(custom_peak_labels):
                display_text = f"{custom_peak_labels[i]}: {peak_energy:.3f} eV"
            else:
                display_text = f"Peak {i+1}: {peak_energy:.3f} eV"

            t = ax_dens.text(0.05, peak_energy + 0.005, display_text, fontsize=8, color='black', verticalalignment='bottom', zorder=11)
            t.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])

    ####################
    if plot_z_vs_e and z_data is not None:
        fig_corr, ax_corr = plt.subplots(figsize=(4, 3))
        # Take the first dataset for the correlation plot
        scat_energies = next(iter(energy_datasets.values()))
        
        ax_corr.scatter(z_data, scat_energies, s=20, alpha=0.7, c = 'white', edgecolors='black', linewidth=0.5)
        ax_corr.set_xlabel(r'$\Delta z$ (Å)', fontsize=fontsize)
        ax_corr.set_ylabel(r'$E_{i}-E_{glob}$ (eV/atom)', fontsize=fontsize)
        ax_corr.set_ylim(e_limit[0], e_limit[1])
        plt.tight_layout()
        plt.savefig(f'{save_path}/z_vs_energy_correlation.png', dpi=300)
    ####################
    
    # Boundary Fixes
    ax_dens.set_ylim(min_e, max_e)
    ax_dens.set_yticks(eticks)
    ax_dens.yaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout()

    # --- Handle Animation Generation ---
    if not plot_density_only and animate_scatter:
        total_points = len(X_eigen)
        
        def update_frame(frame):
            # Calculate dynamic subset slice up to current frame index
            current_count = int((frame + 1) * (total_points / 100)) # 100 frame breakdown step normalization
            current_count = min(max(1, current_count), total_points)
            
            offsets = np.vstack((X_eigen[:current_count], scat_energies[:current_count])).T
            sc.set_offsets(offsets)
            
            if z_data is not None:
                sc.set_array(color_src[:current_count])
            return sc,

        # Create the animator object across 100 key interval steps
        scat_anim = FuncAnimation(fig, update_frame, frames=100, interval=1000//animation_fps, blit=True)
        scat_anim.save(f'{save_path}/{gif_name}', writer='pillow', fps=animation_fps)
        print(f"Animation saved successfully to: {save_path}/{gif_name}")
    else:
        # Standard static save execution
        plt.savefig(f'{save_path}/conf_space.png', dpi=300, bbox_inches='tight')
        
    return fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patheffects as patheffects
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from typing import Optional, List

def plot_structure_landscape(
    X_eigen, 
    energies,                    # Can be an array, list of arrays, or dict of arrays
    z_data=None, 
    show_colorbar=True, 
    cmap='viridis',
    z_limit=(None, None, 5),
    e_limit=(0.0, 0.8, 5),
    figsize=(6, 3),
    show_limits=True,            # Controls automatic peak rendering
    custom_peak_labels: Optional[List[str]] = None,  # NEW PARAMETER: e.g. ["Island", "Flat"]
    fontsize=10, 
    wspace=0.05, 
    cbar_pad=0.02, 
    density_left=True,
    plot_density_only=False,  
    custom_legends=None,         # Explicit text labels list if energies is a list
    density_cmap='Blues',        # Colormap gradient name (e.g., 'Blues', 'Purples', 'magma')
    black_seed_zero=False,       # Force index 0 to be black, remainder to be gradient
    seed_zero_top_zorder=False,  # Force seed 0 to have the highest zorder layer visibility
    fill_density=False,          # Toggle to fill under the curve or not
    density_alpha=0.12,          # Control fill opacity
    dens_line_weight=1.5,
    plot_z_vs_e=False,        # NEW: Option to plot Delta Z vs Energy correlation
    save_path='./'
):
    min_e, max_e = e_limit[0], e_limit[1]
    eticks = np.round(np.linspace(min_e, max_e, e_limit[2]), 1)
    
    # --- Structure & Normalize Input Energies ---
    energy_datasets = {}
    if isinstance(energies, dict):
        energy_datasets = energies
    elif isinstance(energies, (list, tuple)) and len(energies) > 0 and isinstance(energies[0], (list, np.ndarray)):
        for i, dataset in enumerate(energies):
            label = custom_legends[i] if (custom_legends and i < len(custom_legends)) else f"Dataset {i+1}"
            energy_datasets[label] = np.asarray(dataset)
    else:
        # Fallback for single data array
        label = custom_legends[0] if custom_legends else "State Density"
        energy_datasets[label] = np.asarray(energies)

    # --- Setup Gradient Color Palette ---
    num_curves = len(energy_datasets)
    try:
        color_gradient = plt.colormaps.get_cmap(density_cmap)
    except AttributeError:
        color_gradient = plt.cm.get_cmap(density_cmap)

    sampled_colors = []
    if num_curves > 1:
        if black_seed_zero:
            sampled_colors.append('black')
            remaining_count = num_curves - 1
            if remaining_count > 1:
                color_indices = np.linspace(0.85, 0.35, remaining_count)
                for x in color_indices:
                    sampled_colors.append(color_gradient(x))
            else:
                sampled_colors.append(color_gradient(0.60))
        else:
            color_indices = np.linspace(0.95, 0.35, num_curves)
            sampled_colors = [color_gradient(x) for x in color_indices]
    else:
        sampled_colors = ['black' if black_seed_zero else color_gradient(0.85)]

    # --- Setup Subplots Axis Framework ---
    if plot_density_only:
        width_mod = 0.55 if num_curves > 1 else 0.4
        fig, ax_dens = plt.subplots(figsize=(figsize[0] * width_mod, figsize[1]))
        ax_scat = None
        ax_l, ax_r = ax_dens, None
    else:
        ratios = [1, 2.5] if density_left else [2.5, 1]
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={'width_ratios': ratios})
        fig.subplots_adjust(wspace=wspace)
        ax_scat = ax_r if density_left else ax_l
        ax_dens = ax_l if density_left else ax_r

    # --- Scatter Plot Representation ---
    if not plot_density_only:
        if z_data is not None:
            vmin = z_limit[0] if z_limit[0] is not None else np.min(z_data)
            vmax = z_limit[1] if z_limit[1] is not None else np.max(z_data)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            color_src = z_data
        else:
            norm = None
            color_src = 'white'

        scat_energies = next(iter(energy_datasets.values())) if num_curves > 1 else list(energy_datasets.values())[0]

        if len(X_eigen) == len(scat_energies):
            sc = ax_scat.scatter(X_eigen, scat_energies, c=color_src, s=25,
                                 cmap=cmap if z_data is not None else None,
                                 norm=norm, edgecolors='black', linewidth=0.5,
                                 alpha=0.8, zorder=2)
            
            if show_colorbar and z_data is not None:
                cbar = plt.colorbar(sc, ax=ax_scat, pad=cbar_pad)
                cbar.set_label(r'$\Delta z$ (Å)', fontsize=fontsize)
                tick_locs = np.linspace(vmin, vmax, z_limit[2])
                cbar.set_ticks(tick_locs)
                cbar.set_ticklabels([f"{t:.2f}" for t in tick_locs])
        else:
            print("Warning: X_eigen layout dims don't match base layout. Skipping scatter population.")

    # --- Evaluate & Generate Multiple Gaussian KDE Curves with Gradient Colors ---
    energy_grid = np.linspace(min_e, max_e, 200)
    total_density = np.zeros_like(energy_grid)

    for idx, (name, data_array) in enumerate(energy_datasets.items()):
        if len(data_array) > 1:
            kde = gaussian_kde(data_array)
            density = kde.evaluate(energy_grid)
            total_density += density

            curve_color = sampled_colors[idx]

            if idx == 0 and seed_zero_top_zorder:
                current_line_zorder = 10
                current_fill_zorder = 9
            else:
                current_line_zorder = 4
                current_fill_zorder = 3

            # Plot profiles
            ax_dens.plot(density, energy_grid, color=curve_color, lw=dens_line_weight, zorder=current_line_zorder, label=name)

            if fill_density:
                ax_dens.fill_betweenx(energy_grid, 0, density, color=curve_color, alpha=density_alpha, zorder=current_fill_zorder)

    ax_dens.set_xlabel('State Density (config./eV)', fontsize=fontsize)

    if num_curves > 1:
        fig.legend(
            frameon=False,
            fontsize=fontsize-2,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5)
        )

    if not plot_density_only and ax_scat is not None:
        ax_scat.set_xlabel(r'$\psi_{1d}(a.u.)$', fontsize=fontsize)
        ax_scat.xaxis.set_minor_locator(AutoMinorLocator())

    if plot_density_only or density_left:
        ax_dens.set_ylabel(r'$E_{i}-E_{glob}$ (eV/atom)', fontsize=fontsize)
        
    # --- Automatic Peak Detection and Visualization Logic ---
    if show_limits and len(energy_datasets) > 0:
        # Find local maxima on the aggregated density landscape
        peaks, _ = find_peaks(total_density, prominence=np.max(total_density)*0.05)

        # Sort peaks by energy (lowest energy peak first)
        peaks = sorted(peaks, key=lambda idx: energy_grid[idx])

        axes_to_mark = [ax_dens] if plot_density_only else [ax_l, ax_r]

        for i, peak_idx in enumerate(peaks):
            peak_energy = energy_grid[peak_idx]

            # Draw horizontal lines across all relevant active subplots
            for ax in axes_to_mark:
                if ax is not None:
                    ax.axhline(y=peak_energy, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

            # --- Custom Text Mapping Logic ---
            if custom_peak_labels and i < len(custom_peak_labels):
                display_text = f"{custom_peak_labels[i]}: {peak_energy:.3f} eV"
            else:
                display_text = f"Peak {i+1}: {peak_energy:.3f} eV"

            # Place text overlay on the density axis
            t = ax_dens.text(0.05, peak_energy + 0.005, display_text, fontsize=8, color='black', verticalalignment='bottom', zorder=11)
            t.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])

    # Boundary Fixes
    ax_dens.set_ylim(min_e, max_e)
    ax_dens.set_yticks(eticks)
    ax_dens.yaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout()
    plt.savefig(f'{save_path}/conf_space.png', dpi=300, bbox_inches='tight')
    return fig
