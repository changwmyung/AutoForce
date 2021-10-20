from ase.calculators.calculator import Calculator, all_changes
import theforce.distributed as distrib


class LambdaCalculator(Calculator):

    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']  # etc.

    def __init__(self, calc1, calc2, Lambda, logfile='lambda.log', stdout=False, **kwargs):
        """
        calc1, calc2: ase calculator objects
        Lambda: mixing factor in the range [0, 1]
        """
        super().__init__(**kwargs)
        self.calc1 = calc1
        self.calc2 = calc2
        self.Lambda = Lambda
        self.logfile=logfile
        self._logpref = ''
        self.stdout = stdout

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # cautionary, may be unnecessary
        self.calc1.results.clear()
        self.calc2.results.clear()

        # calculate
        self.calc1.calculate(self.atoms)
        self.calc2.calculate(self.atoms)

        # Lambda
        for quantity in self.implemented_properties:
            self.results[quantity] = (self.calc1.results[quantity] * (1. - self.Lambda) +
                                      self.calc2.results[quantity] * self.Lambda)

        du=self.calc2.results['energy']-self.calc1.results['energy']
        self.log('{} {} {}'.format(self.Lambda, du, self.atoms.get_temperature()
                                  ))

    @property
    def rank(self):
        if distrib.is_initialized():
            return distrib.get_rank()
        else:
            return 0

    def log(self, mssge, mode='a'):
        if self.logfile and self.rank==0: 
            with open(self.logfile, mode) as f:
                f.write('{} {}\n'.format(
                    self._logpref, mssge))
                if self.stdout:
                    print('{} {}'.format(
                        self._logpref, mssge))

def test():
    from theforce.calculator.active import ActiveCalculator, kcal_mol
    from theforce.calculator.socketcalc import SocketCalculator
    from theforce.util.parallel import mpi_init
    from ase.calculators.vasp import Vasp
    from ase.build import bulk

    from theforce.calculator.active import ActiveCalculator, FilterDeltas
    from theforce.similarity.sesoap import SeSoapKernel, SubSeSoapKernel
    from theforce.util.parallel import mpi_init
    from theforce.util.aseutil import init_velocities, make_cell_upper_triangular
    from ase.build import bulk
    from ase.md.npt import NPT
    from ase import units

    #
    #common = dict(command='mpirun -n 8 vasp_std')
    from ase.io import read

    process_group = mpi_init()
    common = dict(ediff=0.04, fdiff=0.04, process_group=process_group)

    active_kwargs_1 = {'calculator': None,
                       'covariance': "model_1.pckl"
                      }

    active_kwargs_2 = {'calculator': None,
                       'covariance': "model_2.pckl"
                      }

    calc1 = ActiveCalculator(**active_kwargs_1, **common )
    calc2 = ActiveCalculator(**active_kwargs_2, **common)

    Lambda = 0.1
    calc = LambdaCalculator(calc1, calc2, Lambda)
    #
    atoms=read('POSCAR')
    atoms.calc = calc

    npt = False
    tem = 300.
    stress = 0.
    dt = 0.5
    ttime = 25*units.fs
    ptime = 100*units.fs
    bulk_modulus = 0.0
    pfactor = (ptime**2)*bulk_modulus*units.GPa
    init_velocities(atoms, tem)
    # make_cell_upper_triangular(atoms)
    #filtered = FilterDeltas(atoms)
    dyn = NPT(atoms, dt*units.fs, temperature_K=tem, externalstress=stress*units.GPa,
          ttime=ttime, pfactor=pfactor if npt else None, mask=None, trajectory='md.traj',
          append_trajectory=False, loginterval=1)

    # F. run md
    dyn.run(1000)

    #test
    #energy = atoms.get_potential_energy()
    #forces = atoms.get_forces()
    #print(f'Lambda: {Lambda}   Energy: {energy}   Forces: {forces}')

if __name__ == '__main__':
    test()
