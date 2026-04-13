
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from mpl_toolkits.axes_grid1 import make_axes_locatable
from ase import Atoms
from ase.constraints import FixAtoms

from typing import Optional, Sequence
from agox.utils.plot.colors import Colors
from agox.utils.plot import plot_atoms, plot_cell

def plot_structure(
	atoms: Atoms,
	plane: str = 'yz+',
	radius_factor: float = 0.8,
	repeat: int = 1,
	
	figsize: tuple[int, int] = (6, 5),
	save_path: Optional[str] = None,
	
	darken_factor: float = 0.5,
	constraint_symbols: Sequence[str] = ['Mg', 'O'],
	height_darken_symbols: Sequence[str] = ['Fe'],
	
	num_ticks: int = 5 ,
	cbar_label: str = None,
	linewidths: float = 1.5,
	cell_offset = np.array([0, 0, 0]),
	
	show_colorbar: bool = False,
	add_cell: bool = False,
	plot_show: bool = True,
	axis_off: bool = False,

	fontsize: int = 12,
	max_cbar: float = None,
	
	element_base_colors: Optional[dict] = None,
	default_color: Sequence[float] = [0.5, 0.5, 0.5],

	) -> None:
	
	if constraint_symbols:
		con_indices = [a.index for a in atoms if a.symbol in constraint_symbols]
		atoms.set_constraint(FixAtoms(indices=con_indices))
	
	atom_colors = Colors(atoms)
	
	if element_base_colors is None:
		element_base_colors = {
			'O': np.array([1.0, 0.0, 0.0]),
			'Mg': np.array([1.0, 0.65, 0.0]),
			'Fe': np.array([0.0, 0.8, 0.0])
		}
	
	fig, ax = plt.subplots(figsize=figsize)
	for sym in constraint_symbols:
		indices = [a.index for a in atoms if a.symbol == sym]
		if indices and sym in element_base_colors:
			base_color = element_base_colors[sym]
			atom_colors.set_color(base_color, indices=indices)
	
	last_cmap = None
	norm = None
	
	if height_darken_symbols:
		all_h_indices = [a.index for a in atoms if a.symbol in height_darken_symbols]
		if all_h_indices:
			z_positions = atoms.positions[all_h_indices, 2]
			global_z_min = np.min(z_positions)
			
			if max_cbar:
				global_z_max = max_cbar
			else:
				global_z_max = np.max(z_positions)
			
			norm = Normalize(vmin=0, vmax=global_z_max)
			
			for sym in height_darken_symbols:
				indices = [a.index for a in atoms if a.symbol == sym]
				if not indices or sym in constraint_symbols: continue
				
				base_color = element_base_colors.get(sym, default_color)
				dark_target = base_color * (1 - darken_factor)
				last_cmap = LinearSegmentedColormap.from_list(f'{sym}_map', [base_color, dark_target])
				
				for idx in indices:
					rel_z = atoms.positions[idx, 2] - global_z_min
					color_val = last_cmap(norm(rel_z))
					atom_colors.set_color(color_val, indices=[idx])
			
			plot_atoms(
				ax, atoms, colors=atom_colors, plane=plane,
				radius_factor=radius_factor, plot_constraint=True,
				patch_kwargs=dict(linewidth=1.0), repeat=repeat
			)
			
			if add_cell:
				plot_cell(
					ax,
					atoms.cell,
					plane=plane,
					collection_kwargs=dict(
						linewidths=linewidths,
						linestyles='--',
						dashes=(0, (5, 10)),
					),
				)
			
			if not (np.all(cell_offset == 0)):
				plot_cell(
					ax,
					atoms.cell,
					plane=plane,
					offset=cell_offset,
					collection_kwargs=dict(linewidths=0),
				)
				
				plot_cell(
					ax,
					atoms.cell,
					plane=plane,
					offset=-cell_offset,
					collection_kwargs=dict(linewidths=0),
				)
			
			if axis_off:
				ax.set_axis_off()
			else:
				for spine in ax.spines.values():
					spine.set_linewidth(linewidths)
			
			if show_colorbar:
				divider = make_axes_locatable(ax)
				cax = divider.append_axes("right", size="5%", pad=0.1)
				
				sm = ScalarMappable(norm=norm, cmap=last_cmap)
				cbar = fig.colorbar(sm, cax=cax)
				
				tick_locs = np.linspace(0, norm.vmax, num_ticks) # custom ticks
				cbar.set_ticks(tick_locs)
				cbar.set_ticklabels([f"{t:.2f}" for t in tick_locs])
				cbar.set_label(cbar_label, fontsize=fontsize)
				
				cax.tick_params(
					labelsize=fontsize,
					width=linewidths,
					length=12,	
				)
				
				cbar.outline.set_linewidth(linewidths)
				
			plt.tight_layout()
			if save_path: plt.savefig(save_path, bbox_inches='tight')
			if plot_show: plt.show()
			
			plt.close()

"""
from ase.io import read
cell_offset = np.array([5, 5, 0])

structs = read(f"{dir_out}/{dir_xsf}/19/struct_960.xsf")
atom_center = structs.get_positions().mean(axis=0)
structs.rotate(90, 'z', center=atom_center)

plot_structure(
	structs,
	plane='xy+',
	figsize=(5, 5),
	cell_offset = cell_offset,
	repeat=3,
	constraint_symbols=['Mg', 'O'],
	height_darken_symbols=['Fe'],
	add_cell = True,
	radius_factor = 0.9,
	linewidths = 2.5,
	cbar_label=r'$\Delta z$ ($\AA$)',
	show_colorbar = True,
	fontsize = 21,
	darken_factor=0.7, # 0.0 = original color, 1.0 = goes toward black
	num_ticks=3 # Custom number of cbar axis values
)
"""