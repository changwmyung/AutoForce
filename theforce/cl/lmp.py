# +
from lammps import lammps
from ase.atoms import Atoms
from ase.calculators.lammps import convert
import numpy as np


def read_lammps_file(file):
    commands = []
    units = None
    fixID = None
    fixIndex = None
    scope = {}
    for line in open(file):
        if line.lower().startswith('#autoforce'):
            exec(line[10:].strip(), scope)
            continue
        if '#' in line:
            line = line[:line.index('#')]
        line = ' '.join(line.split())
        if line == '':
            continue
        if line.startswith('units'):
            units = line.split()[1]
        if line.lower().startswith('fix autoforce'):
            fixID = line.split()[1]
            fixIndex = len(commands)
        commands.append(line)
    map_numbers = scope['atomic_numbers']
    if fixID is None:
        raise RuntimeError('no fix autoforce!')
    return units, map_numbers, fixID, fixIndex, commands


def get_cell():
    global lmp
    boxlo, [xhi, yhi, zhi], xy, yz, xz, pbc, box_change = lmp.extract_box()
    cell = np.array([[xhi, xy, xz],
                     [0., yhi, yz],
                     [0., 0., zhi]])
    return cell, pbc


def callback(caller, ntimestep, nlocal, tag, pos, fext):
    global lmp, atoms, units, map_numbers, fixID, nktv2p, calc

    # build atoms
    cell, pbc = get_cell()
    cell = convert(cell, 'distance', units, 'ASE')
    xyz = np.array(lmp.gather_atoms('x', 1, 3)).reshape(-1, 3)
    positions = convert(xyz, 'distance', units, 'ASE')
    if atoms is None:
        numbers = np.array(lmp.gather_atoms('type', 0, 1))
        numbers = list(map(map_numbers.get, numbers))
        atoms = Atoms(numbers=numbers, positions=positions, pbc=pbc, cell=cell)
        atoms.calc = calc
    else:
        atoms.cell = cell
        atoms.positions = positions

    # calculate energy, force, and virial
    f = atoms.get_forces()[tag-1]
    e = atoms.get_potential_energy()
    fext[:] = convert(f, 'force', 'ASE', units)
    e = convert(e, 'energy', 'ASE', units)
    lmp.fix_external_set_energy_global(fixID, e)
    if 'stress' in atoms.calc.implemented_properties:
        v = atoms.get_stress()
        v = convert(v, 'pressure', 'ASE', units)
        vol = atoms.get_volume()
        v = -v / (nktv2p[units]/vol)
        v[3:] = v[3:][::-1]
        lmp.fix_external_set_virial_global(fixID, v)


nktv2p = {"lj": 1.0,
          "real": 68568.415,
          "metal": 1.6021765e6,
          "si": 1.0,
          "cgs": 1.0,
          "electron": 2.94210108e13,
          "micro": 1.0,
          "nano": 1.0,
          }

if __name__ == '__main__':
    # command line args:
    import argparse
    parser = argparse.ArgumentParser(
        description='Dynamics with LAMMPS')
    parser.add_argument('-i', '--input',
                        default='in.lammps', type=str,
                        help='LAMMPS input script')
    args = parser.parse_args()

    # setup lammps
    units, map_numbers, fixID, fixIndex, commands = read_lammps_file(
        args.input)
    lmp = lammps()

    # setup calc
    import theforce.cl as cline
    calc = cline.gen_active_calc()
    atoms = None

    # hide mpi4py from ase! very ugly (see theforce._mpi4py)! TODO: fix this!
    import sys
    del sys.modules['mpi4py']

    # run
    lmp.commands_list(commands[:fixIndex+1])
    lmp.set_fix_external_callback(fixID, callback)
    lmp.commands_list(commands[fixIndex+1:])
