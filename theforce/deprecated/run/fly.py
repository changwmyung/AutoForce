# +
import torch
import numpy as np
from theforce.dynamics.leapfrog import Leapfrog
from theforce.util.aseutil import make_cell_upper_triangular
from theforce.regression.gppotential import PosteriorPotentialFromFolder
import ase.md.velocitydistribution as vd
from ase.md.npt import NPT
from ase.io import read
from ase import units
import os
import string
import glob
import re


def suffix(txt, name, ew=''):
    return re.search(f'{name}_(.+?)'+ew, txt).group(1)


def suffixed_all(name, ew=''):
    return glob.glob(f'{name}_*'+ew)


def suffixed_last(name, ew=''):
    files = suffixed_all(name, ew=ew)
    if len(files) > 0:
        return sorted(files)[-1]
    else:
        return None


def suffixed_new(name, ew=''):
    files = suffixed_all(name, ew=ew)
    new = None
    for suff in string.ascii_lowercase:
        f = f'{name}_{suff}'+ew
        if f not in files:
            new = f
            break
    if new is None:
        raise RuntimeError('ran out of letters!')
    return new, suff


def my_vasp(**kwargs):
    from ase.calculators.vasp import Vasp
    calc = Vasp(command='mpirun -np $NSLOTS vasp_std',
                setups='recommended',
                lreal='Auto',
                istart=0, **kwargs)
    return calc


def clean_vasp():
    vasp_files = ['CHG', 'CHGCAR', 'CONTCAR', 'DOSCAR', 'EIGENVAL', 'OSZICAR', 'OUTCAR',
                  'PCDAT', 'POSCAR', 'POTCAR', 'REPORT', 'vasp.out', 'vasprun.xml', 'WAVECAR',
                  'XDATCAR', 'IBZKPT', 'INCAR', 'KPOINTS']
    other = ['ase-sort.dat', 'gp.chp', 'log.txt']
    junk = ' '.join(vasp_files)+' '+' '.join(other)
    os.system(f'rm -f {junk}')


def clear_all():
    clean_vasp()
    os.system('rm -rf model_*/ md_*.traj leapfrog_*.log strategy.log')


def numpy_same_random_seed():
    seed = torch.tensor(np.random.randint(2**32))
    distrib.broadcast(seed, 0)
    np.random.seed(seed.numpy())


# -

def init_velocities(atoms, t, overwrite=False):
    if atoms.get_velocities() is None or overwrite:
        vd.MaxwellBoltzmannDistribution(atoms, t*units.kB)
        vd.Stationary(atoms)
        vd.ZeroRotation(atoms)


# +
def default_au(n):
    if n == 1:
        return 0.5
    else:
        return 1.


def default_kernel(numbers, cutoff=6., au=None, exponent=4, lmax=3, nmax=3, noise=0.01):
    from theforce.regression.gppotential import GaussianProcessPotential
    from theforce.similarity.heterosoap import HeterogeneousSoapKernel as SOAP
    from theforce.descriptor.cutoff import PolyCut
    from theforce.regression.kernel import White, Positive, DotProd
    from theforce.util.util import date
    kerns = [SOAP(DotProd()**exponent, a, numbers, lmax, nmax,
                  PolyCut(cutoff), atomic_unit=au(a) if au else default_au(a))
             for a in numbers]
    gp = GaussianProcessPotential(
        kerns, noise=White(signal=noise, requires_grad=False))
    with open('gp.chp', 'a') as f:
        f.write(f'# {date()}\n{gp}\n')
    return gp


# -

def strategy(atoms, temperature):
    model = suffixed_last('model', ew='/')
    init_atoms = 'new-atoms'
    if model is None:
        make_cell_upper_triangular(atoms)
    else:
        suff_old = suffix(model, 'model', ew='/')
        if atoms is None:
            atoms = read(f'md_{suff_old}.traj', -1)
            init_atoms = f'resume-md_{suff_old}.traj'
        else:
            make_cell_upper_triangular(atoms)
    init_velocities(atoms, temperature)
    new_model, suff = suffixed_new('model', ew='/')
    traj = f'md_{suff}.traj'
    log = f'leapfrog_{suff}.log'
    with open('strategy.log', 'a') as f:
        f.write(f'{model} -> {new_model}  {init_atoms}  {temperature} Kelvin\n')
    return atoms, model, new_model, traj, log


def fly(temperature, updates, atoms=None, cutoff=None, au=None, calc=None, kern=None, dt=2., tsteps=10,
        ext_stress=0, pfactor=None, mask=None, ediff=0.1, fdiff=0.1, group=None, lf_kwargs={}):
    atoms, model, new_model, traj, log = strategy(atoms, temperature)
    if model is None:
        if not kern:
            kern = default_kernel(np.unique(atoms.numbers), cutoff, au=au)
    else:
        model = PosteriorPotentialFromFolder(model, group=group)
        kern = model.gp
    atoms.set_calculator(calc if calc else my_vasp())
    if hasattr(pfactor, '__iter__'):
        psteps, bulk_modulus = pfactor
        ptime = psteps*dt
        pfactor = (ptime**2)*bulk_modulus
    ase_dyn = NPT(atoms, dt*units.fs, temperature*units.kB, ext_stress,
                  tsteps*dt*units.fs, pfactor, mask=mask, trajectory=traj)
    dyn = Leapfrog(ase_dyn, kern, model=model,
                   algorithm='ultrafast',
                   ediff=ediff, fdiff=fdiff,
                   correct_verlet=False,
                   logfile=log,
                   group=group,
                   **lf_kwargs)
    dyn.run_updates(updates)
    dyn.model.to_folder(new_model)
    if calc is None:
        clean_vasp()
    if group is not None:
        distrib.barrier()
    return dyn.model
