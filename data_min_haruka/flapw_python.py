from pathlib import Path
import re
import numpy as np

from ase.calculators.calculator import FileIOCalculator

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class FLAPW(FileIOCalculator):

    implemented_properties = ["energy"]

    name = "flapw"

    def __init__(
        self,
        command="./flapw",
        input_file="lapwin",
        output_file="lapwout",
        jspins=2,
        star_cutoff=9.8,
        pw_cutoff=3.9,
        kpts=(7, 7, 7),
        smearing=0.001,
        xc="gga",
        starting_state="AFM",
        maxiter=100,
        mixing="A",
        representation="SR",
        **kwargs,
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.jspins = jspins
        self.star_cutoff = star_cutoff
        self.pw_cutoff = pw_cutoff
        self.kpts = kpts
        self.smearing = smearing
        self.xc = xc
        self.starting_state = starting_state
        self.maxiter = maxiter
        self.mixing = mixing
        self.representation = representation

        super().__init__(
            command=command,
            **kwargs,
        )

    def write_input(
        self,
        atoms,
        properties=None,
        system_changes=None,
    ):
        super().write_input(
            atoms,
            properties,
            system_changes,
        )

        infile = Path(self.directory) / self.input_file

        self.write_lapwin(
            atoms,
            infile,
        )

    def write_lapwin(
        self,
        atoms,
        filename,
    ):
        formula = atoms.get_chemical_formula()

        ANG_TO_BOHR = 1.889726125

        cell_bohr = atoms.cell.array * ANG_TO_BOHR

        lattice_vectors = cell_bohr.copy()

        symbols = atoms.get_chemical_symbols()
        
        frac_coords = atoms.get_scaled_positions()

        with open(filename, "w") as f:

            f.write(
                f"Title: {formula}\n"
            )

            f.write(
                "Mode: bulk  auto                               !bulk/film auto/kpts/base\n"
            )

            f.write(
                "*** lattice vectors **********************\n"
            )

            f.write("1.00000\n")

            for vec in lattice_vectors:

                f.write(
                    f"{vec[0]:12.6f} "
                    f"{vec[1]:12.6f} "
                    f"{vec[2]:12.6f}\n"
                )

            f.write(
                "*** atomic number cartesian or internal coordinates\n"
            )

            f.write(
                "set pos file:F, name:fconst_pos1.dat\n"
            )

            f.write("internal\n")

            prev_symbol = None

            for sym, pos in zip(
                symbols,
                frac_coords,
            ):

                x, y, z = pos

                if sym != prev_symbol:

                    f.write(
                        f"{sym:<2} "
                        f"{x:12.8f} "
                        f"{y:12.8f} "
                        f"{z:12.8f}\n"
                    )

                    prev_symbol = sym

                else:

                    f.write(
                        f"   "
                        f"{x:12.8f} "
                        f"{y:12.8f} "
                        f"{z:12.8f}\n"
                    )

            #space and general option
            f.write("*** SPACE GROUP ***************************\n")
            f.write(f"Representation:{self.representation} ,option:default              !SR/ZR/FR/UR SR:default\n")
            f.write("*** GENERAL OPTIONS ***********************\n")
            f.write("Density of states:F, option:default\n")
            f.write("Band structure:F, option:default\n")
            f.write("Density plot:F, option:default\n")
            f.write("Slice analysis:F, option:set\n")
            f.write("Force calculation:F, option:default\n")
            f.write("Geometry optimization:F, option:default\n")
            f.write("Nudged elastic band:F, option:default\n")
            f.write("Force constant calculation:F, option:default\n")
            f.write("Electron-phonon coupling:F, option:set\n")
            f.write("External E field:F, option:set\n")
            f.write("External H field:F, option:set\n")
            f.write("Jellium potential:F, option:set\n")
            f.write("Electric field gradient:F, option:default\n")
            f.write("L matrix calculation:F, option:default\n")
            f.write("P matrix calculation:F, option:default\n")
            f.write("J matrix calculation:F, option:default\n")
            f.write("Second variational +U:F, option:set\n")
            f.write("Second variational SOC:F, option:default\n")
            f.write("Dispersion correction:F, option:default\n")
            f.write("Magnetic dipole-dipole:F, option:default\n")
            f.write("Noncollinear Magnetism:F, option:set\n")
            f.write("Equi-density constraint:F, option:set\n")

            rmt_table = {}

            with open("README_MT-default") as h:
                for line in h:

                    m = re.search(
                        r"element\('([A-Za-z ]+)'\s*,\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*,\s*([0-9.]+)",
                        line
                    )

                    if m:
                        symbol = m.group(1).strip()
                        lmax = int(m.group(2))
                        rmt = float(m.group(3))

                        rmt_table[symbol] = {
                            "lmax": lmax,
                            "rmt": rmt
                        }

            species = []

            for s in atoms.get_chemical_symbols():
                if s not in species:
                    species.append(s)

            f.write("*** BASES *********************************\n")
            f.write(f"star-function cut-off:{self.star_cutoff}\n")
            f.write(f"jspins={self.jspins}\n")
            f.write("e_float:T, xo:T\n")
            f.write("nwin=1\n")
            f.write(f"plane-wave cut off:{self.pw_cutoff}\n")
            f.write("number of states:0\n")
            f.write("lapw parameters:set\n")

            for elem in species:
                rmt = rmt_table[elem]["rmt"]
                lmax = rmt_table[elem]["lmax"]

                f.write(f"{elem:<2} rmt={rmt:.2f} lmax={lmax} 0.\n")
                f.write("                   0.\n")
            
            #k-points and mixing option
            f.write("*** K-POINTS ******************************\n")
            f.write("k-point generator:S, option:default\n")
            f.write("Smearing:G, parameter:0.001\n")
            f.write("Time-reversal symmetry:T\n")
            f.write("Division along internal axis (each window new line)\n")
            f.write(f"  {self.kpts[0]}   {self.kpts[1]}   {self.kpts[2]}\n")

            f.write("*** MIXING  OPTIONS ***********************\n")
            f.write("(B)royden or (S)traight mixing for density:A\n")
            f.write("Maximum number of iterations:100  20\n")
            f.write("Mixing parameter:0.\n")
            f.write("Convergency: 0.\n")
            #spin_change
            f.write("*** OPTIONS FOR SPIN-POLARIZED CASE *******\n")
            f.write("Spin-options:set\n")
            f.write("Initial spin polarization:T\n")
            f.write(f"Starting state and values:{self.starting_state}\n")

            mag_init = {
                "Fe": 3.0,
                "Co": 3.0,
                "Ni": 2.0,
                "Mn": 4.0,
                "Cr": 3.0,
            }

            species = []

            for sym in atoms.get_chemical_symbols():
                if sym not in species:
                    species.append(sym)

            for sym in species:
                moment = mag_init.get(sym, 0.0)
                f.write(f" {sym:<2}  {moment:4.1f}    0.0   0.0    0.0\n")
            
            f.write("Mixing parameter: 0.\n")

            f.write("*** ADVANCED SETTINGS *********************\n")
            f.write("Advanced setup:set\n")
            f.write("Output:redu\n")
            f.write("Check potential and density:F\n")
            f.write(f"Exchange correlation:{self.xc}\n")
            f.write("frcor:F, ctail:F\n")

            f.write("*** END ***********************************\n")

    
    def read_results(self):

        outfile = (
            Path(self.directory)
            / self.output_file
        )

        text = outfile.read_text()

        matches = re.findall(
            r"total energy for it=\s*\d+:\s*([-0-9.Ee+]+)\s*htr",
            text,
            flags=re.IGNORECASE,
        )

        if not matches:
            raise RuntimeError(
                "Could not find total energy in lapwout"
            )

        energy_hartree = float(matches[-1])

        HARTREE_TO_EV = 27.211386245988

        self.results["energy"] = (
            energy_hartree * HARTREE_TO_EV
        )
    
    

