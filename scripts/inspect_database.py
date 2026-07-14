from typing import Optional
import numpy as np
from agox.databases import Database

# === Database Inspection Functions ===
def inspect_database(db_file_path: str, start_iter: Optional[int] = None, end_iter: Optional[int] = None):
    """
    Loads an AGOX database and iterates over stored structures 
    filtered between iteration A (start_iter) and iteration B (end_iter).
    """
    print(f"Connecting to database: {db_file_path}\n")
    db = Database(filename=str(db_file_path))
    
    # Extract data
    all_structures_raw = db.get_all_structures_data()
    if not all_structures_raw:
        print("The database is currently empty or could not be read.")
        return

    # Filter iterations
    if start_iter is not None:
        all_structures_raw = [d for d in all_structures_raw if d.get('iteration', 0) >= start_iter]
    if end_iter is not None:
        all_structures_raw = [d for d in all_structures_raw if d.get('iteration', 0) <= end_iter]

    if not all_structures_raw:
        print(f"No structures found within the iteration range [{start_iter} to {end_iter}].")
        return

    # Display summary statistics
    range_str = f"from iteration {start_iter} to {end_iter}" if (start_iter or end_iter) else "all iterations"
    print(f"Total structures found ({range_str}): {len(all_structures_raw)}")
    print('-' * 60)
    print(f"{'Structure ID':<14} | {'Iteration':<10} | {'Energy (eV)':<12} | {'Number of Atoms':<15}")
    print('-' * 60)

    # Print individual row entries
    for raw_data in all_structures_raw:
        struct_id = raw_data.get('id')
        iteration = raw_data.get('iteration')
        energy = raw_data.get('energy')
        
        # 'type' contains the atomic numbers array (e.g., [6, 6, 1, 1] for C2H2)
        atomic_numbers = raw_data.get('type') 
        num_atoms = len(atomic_numbers) if atomic_numbers is not None else 0
        
        print(f"{struct_id:<14} | {iteration:<10} | {energy:<12.4f} | {num_atoms:<15}")

    print('-' * 60)
    
    # Convert raw structures to active ASE Atoms objects for deep geometry inspections
    print("\nConverting rows to ASE Atoms for deep geometry inspections...")
    for raw_data in all_structures_raw[:3]:  # Inspecting the first three matching the filter
        struct_id = raw_data.get('id')
        iteration = raw_data.get('iteration')
        atoms_obj = db.db_to_atoms(raw_data)
        
        print(f"\n--- Checking Details for Structure ID: {struct_id} (Iteration {iteration}) ---")
        print(f"Chemical Formula: {atoms_obj.get_chemical_formula()}")
        print(f"Positions (First two atoms):\n{atoms_obj.positions[:2]}")
        print(f"Calculated Potential Energy: {atoms_obj.get_potential_energy()} eV")
        
    # Fetch the best structure within this filtered subset
    print("\n" + "=" * 60)
    energies = [c['energy'] for c in all_structures_raw]
    min_idx = energies.index(min(energies))
    best_filtered_atoms = db.db_to_atoms(all_structures_raw[min_idx])
    
    print(f"Best Structure Energy within range [{start_iter} - {end_iter}]: {best_filtered_atoms.get_potential_energy()} eV")
    print(f"Composition: {best_filtered_atoms.get_chemical_formula()}")
    print("=" * 60)