# +
import theforce.cl as cline
from theforce.calculator.active import FilterDeltas
from theforce.util.aseutil import init_velocities, make_cell_upper_triangular
from ase.md.npt import NPT
from ase.io import read
from ase import units
import numpy as np
import os


def md(atoms, dt=None, tem=300., picos=100, bulk_modulus=None, stress=0., mask=None,
       trajectory='md.traj', loginterval=1, append=False, rattle=0.0):
    """
    atoms:        ASE atoms
    dt:           time-step in fs
    tem:          temperature in Kelvin
    picos:        pico-seconds for md
    bulk_modulus: bulk_modulus for NPT simulations. if None, NVT is performed
    stress:       external stress (GPa) for NPT
    mask:         see ase.npt.NPT
    trajectory:   traj file name
    loginterval:  for traj file
    append:       append to traj file
    rattle:       rattle atoms at initial step (recommended ~0.05)
    """

    calc = cline.gen_active_calc()
    atoms.rattle(rattle, rng=np.random)
    atoms.set_calculator(calc)

    # define and run Nose-Hoover dynamics
    if dt is None:
        if (atoms.numbers == 1).any():
            dt = 0.25
        else:
            dt = 1.
    ttime = 25*units.fs
    ptime = 100*units.fs
    if bulk_modulus:
        pfactor = (ptime**2)*bulk_modulus*units.GPa
    else:
        pfactor = None
    init_velocities(atoms, tem)
    make_cell_upper_triangular(atoms)
    filtered = FilterDeltas(atoms)
    steps_for_1ps = int(1000/dt)
    dyn = NPT(filtered, dt*units.fs, tem*units.kB, stress*units.GPa,
              ttime, pfactor, mask=mask, trajectory=trajectory,
              append_trajectory=append, loginterval=loginterval)
    dyn.run(picos*steps_for_1ps)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Machine Learning Molecular Dynamics (MLMD)')
    parser.add_argument('-i', '--input', default='POSCAR', type=str,
                        help='the initial coordinates of atoms, POSCAR, xyz, cif, etc.')
    args = parser.parse_args()
    atoms = read(args.input)
    kwargs = cline.get_default_args(md)
    cline.update_args(kwargs)
    md(atoms, **kwargs)