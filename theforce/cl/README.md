<!-- #region -->
### On-the-fly machine learning from command line
Here we describe the steps required for the machine learning accelerated 
molecular dynamics (MLMD) using VASP, Gaussian, etc from the command line.
For this, we create the usual input input files required for a each 
ab initio software.
Then ML and MD related parameters, which do not depend on underlying 
ab initio calculator, are set in the `ARGS` file.

Running examples are available at top-level
[examples](https://github.com/amirhajibabaei/AutoForce/tree/master/examples) folder.

#### Ab initio settings
The input files for ab initio calculations depend on the software:
* VASP: see [theforce/cl/README_vasp.md](https://github.com/amirhajibabaei/AutoForce/tree/master/theforce/cl/README_vasp.md).
* Gaussian: see [theforce/cl/README_gaussian.md](https://github.com/amirhajibabaei/AutoForce/tree/master/theforce/cl/README_gaussian.md).

Note that these files should be prepared only for single-point energy and force calculations.

Alternatively, one can define a custom calculator using a python script.
In that case, a variable named `calc` should be defined
(see [theforce/calculator/README.md](https://github.com/amirhajibabaei/AutoForce/blob/master/theforce/calculator/README.md), 
section *Parallelism*).

#### Parameters for ML (file=`ARGS`)
The parameters for ML should be provided in a file named `ARGS`. 
This group of parameters are used for setting up the machine
learning calculator and are the same for most ML tasks.
Keep in mind that the default behavior is set such that
the result model of previous calculations is automatically 
loaded if the respective files are present in the working directory.
The following tags are available
```
# inputs
covariance:      'pckl', None, a kernal, folder-name for loading a pickled model (default='pckl')
kernel_kw:       e.g. {'cutoff': 6.}
calculator:      None, 'VASP', 'Gaussian' , or a custom python script (default=None)

# outputs
logfile:         file name for logging (default='active.log')
pckl:            folder-name for pickling the ML model (default='model.pckl')
tape:            for saving all of the model updates (default='model.sgpr')
test:            integer; single-point testing intervals (default=None)

# sampling and optimization
ediff:     (eV)  energy sensitivity for sampling LCEs (default ~ 2 kcal/mol)
fdiff:    (eV/A) forces sensitivity for sampling Ab initio data (default ~ 3 kcal/mol)
noise_f:  (ev/A) bias noise for forces (default ~ 1 kcal/mol)
ioptim:          for hyper-parameter optimization frequency (default=1)
max_data:        max data size (default=inf)
max_inducing:    max inducing size (default=inf)
```
Note that these parameters are not related to any 
ab initio software despite possible name similarities.
They are explained in detail in
[theforce/calculator/README.md](https://github.com/amirhajibabaei/AutoForce/tree/master/theforce/calculator).
The only difference is that here we specify the 
calculator by its name (e.g. `calculator='VASP'`).
For a brief recap read the following.

By default the ML model will be saved in a folder
specified by `pckl` after every update.
For loading the saved model in next simulations
put the following line in `ARGS` after the first 
simulation has ended
```
covariance = 'model.pckl'
```
Thus `covariance` is the input model and `pckl`
is the output model.
The default is `covariance='pckl'` which means
input and output models are the same.
Thus the training is resumed in consecutive runs 
(the model is automatically loaded and saved in
the `pckl` folder).

At the beginning of training, if no covariance
is given (or `= None`), the default kernel will be
instantiated. 
In this case `kernel_kw` can be used for
defining kernel parameters (e.g. `kernel_kw = {'cutoff' : 6.}`).

After sufficient training, one might want to use
the result ML potential for fast simulations 
without further training.
For this, all you need to do is to not specify any 
`calculator` in `ARGS` or set `calculator=None`.
In this case, the configurations for which
the uncertainty is large will be saved in the
`active_uncertain.traj` file which are excellent
candidates for offline training.

The other paramteres (`ediff`, `fdiff`) which control 
the sampling and accuracy should be gradually tuned to 
get the desired accuracy.

If `ioptim = 0`, hyper-parameters are (re)optimized
for every data/LCE increament. This can dramatically
slow down MLMD. If `ioptim = 1` the model is
(re)optimized once for every step that
total increaments are nonzero. If `ioptim = i`
where i>1, the model is (re)optimized only
when n.o. sampled data is divisible by (i-1).

If the size of accumulated data/inducing becomes too large,
the simulation may become too slow.
`max_data` and `max_inducing` can be used to set an
upper limit for these sizes.


#### Model initialization from random displacements (optional)
In most cases starting from an empty ML model 
is alright and the model can be built from scratch
on-the-fly with dynamics.
In some sensitive cases it could be difficult for
the dynamics to get started smoothly.
Too many atom types, too small systems, and highly 
unstable arrangement of atoms are examples of such 
cases.
Also presence of hydrogen is always challenging.
For such cases it could be helpful to generate a
preliminary model by a few random displacements 
of atoms from their initial positions.
For this we can execute `init_model` (see **Run** in proceeding)
where the following tages can be set in the `ARGS` file
```
samples:      number of samples (default=5)
rattle:       stdev for random displacements (default=0.05)
trajectory:   traj file name (defult='init.traj')
```
After execution a pickled model will be generated
in the working directory (default name is `pckl='model.pckl/'`).

#### Parameters for MD (file=`ARGS`)
The parameters for MD are also set in the `ARGS` file.
The following tags are available
```
dt:           time-step in fs (default=1. or 0.25 if H is present)
dynamics:     'NPT', 'Langevin' (default='NPT')
tem:          temperature in Kelvin (default=300.)
picos:        pico-seconds for md (default=100)
bulk_modulus: bulk_modulus for NPT simulations. if None (default), NVT is performed
stress:       external stress (GPa) for NPT (default=0.)
mask:         see ase.npt.NPT (default=None)
iso:          if True (or 1), keep the shape constant (default=False)
trajectory:   traj file name (default='md.traj')
loginterval:  for traj file (default=1)
append:       append to traj file (default=False)
rattle:       rattle atoms at the initial step (default=0.0)
tdamp:        temperature damping time (fs) (default=25)
pdamp:        pressure damping time (fs) (default=100)
friction:     for Langevin dynamics (default=1e-3)
ml_filter:    range(0, 1); filters force discontinuities due to ML updates (default=0.8)
```
All of the above tags have default values which will be overridden
with the settings in `ARGS`. 
A minimal `ARGS` file could contain the following
```
calculator = 'VASP'
tem = 300.
picos = 20
```

The default timestep is `dt=1.` femtosecond 
(`dt=0.25` if hydrogen is present) which is 
intentionally smaller than typical `dt` for AIMD
because updating the model on-the-fly causes 
discontinuities in the forces.
Smaller `dt` increases the stability in these cases.
If the ML model is mature, larger `dt` can be used
(even larger than AIMD).

A temperature list is also accepted in which
case the MD will be continued at all given
temperatures. For example
```
tem = [1000., 1100., 1200., 1300., 1400., 1500.]
# or tem = arange(1000., 1500., 100.)
# or tem = linspace(1000., 1500., 6)
```

If `bulk_modulus` is given, NPT simulation
will be performed (with cell fluctuations).
It is recommended to first perform a NVT
simulation and then use the result `pckl`
as the starting potential for the NPT simulation.
NPT simulations are more vulnerable to sudden ML updates
when starting from an empty model.

A practical issue may rise if the MD simulation starts 
with an empty model and a state with forces very close to zero.
If the initial configuration doesn't have diverse
local environments (e.g. perfect crystaline strutures),
the predictive variance remains very close to 0 and
the active learning algorithm fails.
For this, we have introduced the `rattle` tag 
which disturbs the atoms at the initial state.
The default value is `rattle=0.0`.

If hydrogen is present in the system, 
faster kinetic damping (smaller `tdamp`) may be needed, 
at least in the beginning of training.
In general faster damping leads to smoother training.

`ml_filter` is used for smooth variation of forces 
when the ML model is updated. The sudden changes of
forces will appear as residual force which shrinks
every step by a factor given by `ml_filter`.
This smoothens the dynamics but it can also lead to
bad dynamics (bond breaking, etc) if the frequency 
of ML updates is too much.
It can be deactivated by setting it to `0.` or `None`.

#### Parameters for structure relaxation (file=`ARGS`)
The parameters for structure relaxation (minimization of forces)
are also set in the `ARGS` file. The following tags are available
```
fmax:         maximum forces (default=0.01)
cell:         if True, minimize stress too (default=False)
mask:         stress components for relaxation (default=None)
algo:         algo from ase.optimize (default='BFGS')
trajectory:   traj file name (default='relax.traj')
rattle:       rattle atoms at initial step (default=0.02)
clear_hist:   if true, clear optimizer history when ML model is updated (default=False)
confirm:      if True, Ab initio for the last step and potentially reoptimize (default=True)
```

Other possible entries for `algo` are: `'LBFGS', 'GPMin', 'FIRE', 'MDMin'`
(see [this](https://wiki.fysik.dtu.dk/ase/ase/optimize.html)).

`rattle` is the stdev for random displacement of atoms
at the initial step.
This is extremely beneficial for ML if the initial
structure is ordered and the local environments of 
many atoms are identical.
`rattle` causes some disorder/variance in these environments.

It should be noted that, although the final structure is
fully relaxed according to ML, residual Ab initio
forces will probably be (slightly) larger than `fmax`
(due to ML errors).

#### Parameters for NEB (file=`ARGS`)
```
fmax:         maximum forces (default=0.01)
climb:        for climbing image algorithm (default=False)
algo:         optim algo for NEB (default='BFGS')
algo_if:      optim algo for optimization of the first and last images (default='BFGS')
trajectory:   traj file name for NEB path (default='neb-path.traj')
output:       traj file name for the optimized band (default='neb-out.traj')
```
Relaxation of the first and last images is integrated with NEB.
The in between images can be explicitly provided or generated automatically
(see an [example](https://github.com/amirhajibabaei/AutoForce/tree/master/examples/nudged-elastic-band))

#### On setting constraints
The best option for setting constraints is to use ASE's
python inteface for `Atoms`
(see [this](https://wiki.fysik.dtu.dk/ase/ase/constraints.html#module-ase.constraints)).
By saving `Atoms` in `ase.io.Trajectory` files,
the constraints will be preserved when reading.
Thus the traj files can be used for passing initial
coodinates (e.g. `-i ini.traj`) along with constraints
for MD, relaxation or NEB.


#### Run
Lets assume that 20 cores are available.
We split these cores to 12 for the underlying ab initio 
software and 8 for ML.
Specification of cores for the ab initio software 
is done in their respective input files (or in a file named `COMMAND`).
Then, the simulation can be started with 
the following script
```sh
python -m theforce.calculator.calc_server &
sleep 1 # waits 1 sec for the server to be set
#
### choose one of following depending on the task
# mpirun -n 8 python -m theforce.cl.init_model  # for model initialization
# mpirun -n 8 python -m theforce.cl.relax       # for structure relaxation
mpirun -n 8 python -m theforce.cl.md            # for ML accelerated MD
```
Optionally, input and ouput atomic positions can be specified by
`-i input-name` and `-o output-name`.

#### Outputs
The trajectory is saved in `trajectory='md.traj'` for MD
and `'relax.traj'` for relaxation by default.
This file can be processed using the `ase.io` module.
The final coordinates of atoms are also written
in some capacity for convenience.
Possibly a folder named `vasp` or `gaussian_wd` which contains
the most recent (single-point) ab initio calculation
may be present.
The log for ML is written in `logfile='active.log'` which 
can be visualized (converted to `active.pdf`) by
```sh
python -m theforce.calculator.active active.log
```
The resulting ML model is saved in the `pckl` 
(default=`'model.pckl'`) which can be used as the 
input model for the next simulation.
ML updates are saved in the file specified by `tape`
(default=`'model.sgpr'`).
This file can be used for retraining with different
parameters, etc.

#### Post-training
After the training is finished, we can perform MD
using only the ML potential (without further updates) simply by
```sh
mpirun -n 20 python -m theforce.cl.md -i POSCAR  # or -i Gaussian.gjf, coords.xyz, etc.
```
provided that `ARGS` has the following entries
```
covariance = `model.pckl` # path to the saved model
calculator = None         # not necessary because both are defaults
```
This time all 20 cores are used for ML and we can simulate
much bigger systems.

### Training with existing data
A utility command is provided for training with existing Ab initio data.
For this, set the appropriate parameters for ML in `ARGS`.
Then refer to the following patterns.
```sh
mpirun -n 20 python -m theforce.cl.train -i data.traj           # read all data
mpirun -n 20 python -m theforce.cl.train -i data.traj -r 0:10:2 # read only 0, 2, ..., 10
mpirun -n 20 python -m theforce.cl.train -i OUTCAR              # read all data
mpirun -n 20 python -m theforce.cl.train -i OUTCAR -r :-1:      # read only the last one
mpirun -n 20 python -m theforce.cl.train -i OUTCAR-1 OUTCAR-2   # read two OUTCAR files
mpirun -n 20 python -m theforce.cl.train -i */OUTCAR            # all OUTCAR files in subfolders
mpirun -n 20 python -m theforce.cl.train -i model.sgpr          # use the tape from another training
```
Similarly, for testing
```sh
mpirun -n 20 python -m theforce.cl.test -i dft.traj -o ml.traj # optionally -r ::
python -m theforce.regression.scores ml.traj dft.traj          # prints a description of errors   
```

### On `.sgpr`, `.pckl` files and checkpointing

During on-the-fly training, ML updates are saved in two forms:
`model.pckl` and `model.sgpr`. 
One can change the names of these files in `ARGS` via `pckl` 
and `tape` keywords.
These two files contain essentially the same information 
with a difference that `.sgpr` contains only the minimal
raw data in textual form while `.pckl` contains the 
calculated descriptors, regression matrices, etc.
as well in the binary format.
If one deletes a `.pckl` file since it uses a lot of memory, 
it can easily be regenerated from `.sgpr` via the `build` 
command (non-default filenames should be specified in `ARGS`):
```sh
[mpirun -np ...] python -m theforce.cl.build
```
Thereforce `.sgpr` files are suitable for sharing
SGPR models and long-term storage.

Additionally, a functionality is built into `.sgpr` files
which allows them to import other `.sgpr` files using
the `include` keyword.
A line like
```
include: path/to/another/x.sgpr
```
in `.sgpr` has the same effect as copying the content of
`x.sgpr` here where the path can be either relative or absolute.
Note that it is allowed for `x.sgpr` to import other `.sgpr` 
files recursively.
In recursive imports,
one does not need to worry about duplicate imports
since a list of absolute paths for the imported files
are stored internally and duplicate occurrences of
a file are eliminated.
This allows modular storage of training stages which
can be extreemly usefull for complex projects.
For instance if there is a list of trained models
stored in `~/.sgpr/` folder (which can be linked to 
each other by `include` commands), one can create
`model.sgpr` in the working directory and input
```
include: ~/.sgpr/a.sgpr
include: ~/.sgpr/b.sgpr
include: ~/.sgpr/c.sgpr
...
```
Then a `.pckl` can be generated using the `build` command
which can be used a the initial model for other training steps.

In addition to `build`, such `.sgpr` files can be used
by the `train` command.
The difference is that `build` includes all of the data
which are read from `.sgpr` files while `train` only
samples from those data where the sampling can be controlled
using the `ediff` and `fdiff` keywords.
<!-- #endregion -->
