# +
from ase.calculators.vasp import Vasp

calc = Vasp(command="mpirun -n 4 vasp_std", xc='pbe',
            directory='vasp'
            )
