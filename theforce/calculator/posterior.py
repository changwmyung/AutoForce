# +
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
import torch
from torch.autograd import grad
import warnings
import ase
from theforce.descriptor.atoms import TorchAtoms
import socket
from ase.io import read
import os


class PosteriorCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'free_energy']

    def __init__(self, potential, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.potential = potential

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.atoms.update(forced=True)
        # energy and forces
        energy = self.potential([self.atoms], 'energy',
                                all_reduce=self.atoms.is_distributed)
        forces = self.potential([self.atoms], 'forces',
                                all_reduce=self.atoms.is_distributed)
        # results
        self.results['energy'] = energy.detach().numpy()[0]
        self.results['forces'] = forces.detach().numpy()
        # NOTE: check if this is correct!
        self.results['free_energy'] = self.results['energy']


class PosteriorStressCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'free_energy', 'stress']

    def __init__(self, potential, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.potential = potential

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        if self.potential.is_distributed:
            raise NotImplementedError(
                '(Auto)stress in distributed mode is not implemented')
        Calculator.calculate(self, atoms, properties, system_changes)
        self.atoms.update(cellgrad=True, forced=True)
        # energy and forces
        energy = self.potential([self.atoms], 'energy', enable_grad=True,
                                all_reduce=self.atoms.is_distributed)
        forces = self.potential([self.atoms], 'forces',
                                all_reduce=self.atoms.is_distributed)
        # stress
        stress1 = -(forces[:, None]*self.atoms.xyz[..., None]).sum(dim=0)
        cellgrad, = grad(energy, self.atoms.lll)
        if self.atoms.is_distributed:
            torch.distributed.all_reduce(cellgrad)
        stress2 = (cellgrad[:, None]*self.atoms.lll[..., None]).sum(dim=0)
        stress = (stress1 + stress2).detach().numpy() / self.atoms.get_volume()
        # results
        self.results['energy'] = energy.detach().numpy()[0]
        self.results['forces'] = forces.detach().numpy()
        self.results['free_energy'] = self.results['energy']
        self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]


class PosteriorVarianceCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'free_energy']

    def __init__(self, potential, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.potential = potential

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if self.atoms.is_distributed:
            raise NotImplementedError(
                'variance is not implemented for distributed atoms')
        self.atoms.update()
        energy, energy_var = self.potential(
            [self.atoms], 'energy', variance=True)
        forces, forces_var = self.potential(
            [self.atoms], 'forces', variance=True)
        self.results['energy'] = energy.detach().numpy()[0]
        self.results['forces'] = forces.detach().numpy()
        # variances
        self.results['energy_var'] = energy_var.detach().numpy()[0]
        self.results['forces_var'] = forces_var.detach().numpy()
        # NOTE: check if this is correct!
        self.results['free_energy'] = self.results['energy']


class SocketCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, ip='localhost', port=6666, script=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.ip = ip
        self.port = port
        self.script = script

    def ping(self):
        s = socket.socket()
        s.connect((self.ip, self.port))
        s.send(b'?')
        print(f'server says: {s.recv(1024)}')
        s.close()

    @property
    def is_distributed(self):
        return torch.distributed.is_initialized()

    @property
    def rank(self):
        if self.is_distributed:
            return torch.distributed.get_rank()
        else:
            return 0

    @property
    def message(self):
        cwd = os.getcwd()
        msg = f'{cwd}/socket_send.xyz:{cwd}/socket_recv.xyz'
        if self.script is not None:
            msg += f':{cwd}/{self.script}'
        return msg

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        # write and send request
        if self.rank == 0:
            s = socket.socket()
            s.connect((self.ip, self.port))
            self.atoms.write('socket_send.xyz')
            cwd = os.getcwd()
            s.send(self.message.encode())
            assert int(s.recv(1024).decode('utf-8')) == 0
            s.close()
        if self.is_distributed:
            torch.distributed.barrier()
        # read
        atms = read('socket_recv.xyz')
        self.results['energy'] = atms.get_potential_energy()
        self.results['forces'] = atms.get_forces()
        self.results['stress'] = atms.get_stress()

    def close(self):
        s = socket.socket()
        s.connect((self.ip, self.port))
        s.send(b'end')
        s.close()


class AutoForceCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'free_energy', 'stress']

    def __init__(self, potential, variance=False, process_group=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.potential = potential
        self.variance = variance
        self.process_group = process_group

    def calculate(self, _atoms=None, properties=['energy'], system_changes=all_changes):
        if self.potential.is_distributed:
            raise NotImplementedError(
                '(Auto)forces in distributed mode is not implemented')
        if type(_atoms) == ase.atoms.Atoms:
            atoms = TorchAtoms(ase_atoms=_atoms)
            uargs = {'cutoff': self.potential._cutoff,
                     'descriptors': self.potential.gp.kern.kernels}
        else:
            atoms = _atoms
            uargs = {}
        if _atoms is not None and self.process_group is not None:
            atoms.attach_process_group(self.process_group)
        Calculator.calculate(self, atoms, properties, system_changes)
        self.atoms.update(posgrad=True, cellgrad=True,
                          forced=True, dont_save_grads=True, **uargs)
        # energy
        energy = self.potential([self.atoms], 'energy', enable_grad=True, variance=self.variance,
                                all_reduce=self.atoms.is_distributed)
        if self.variance:
            energy, variance = energy
        # forces
        rgrad = grad(energy, self.atoms.xyz, retain_graph=True,
                     allow_unused=True)[0]
        forces = torch.zeros_like(self.atoms.xyz) if rgrad is None else -rgrad
        if self.atoms.is_distributed:
            torch.distributed.all_reduce(forces)
        # stress
        stress1 = -(forces[:, None]*self.atoms.xyz[..., None]).sum(dim=0)
        cellgrad, = grad(energy, self.atoms.lll, allow_unused=True)
        if cellgrad is None:
            cellgrad = torch.zeros_like(self.atoms.lll)
        if self.atoms.is_distributed:
            torch.distributed.all_reduce(cellgrad)
        stress2 = (cellgrad[:, None]*self.atoms.lll[..., None]).sum(dim=0)
        try:
            volume = self.atoms.get_volume()
        except ValueError:
            volume = -2  # here stress2=0, thus trace(stress) = virial (?)
        stress = (stress1 + stress2).detach().numpy() / volume
        # results
        self.results['energy'] = energy.detach().numpy()[0]
        self.results['forces'] = forces.detach().numpy()
        self.results['free_energy'] = self.results['energy']
        self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
        if self.variance:
            self.results['energy_variance'] = variance
