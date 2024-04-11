import numpy as np
from math import exp, sqrt
from numba import int32, float64, types
from numba.experimental import jitclass
from numba import njit
import matplotlib.pyplot as plt

from module import Utils

# the VMC class have the following attributes:
# - system: the quantum system
# - walker: the metropolis walker
# - warmup_steps: number of warmup steps
# - run_steps: number of steps
# - params: the variational parameters
# - plot: verbosity, if True print the acceptance rate and plot
# - warmup_chain: warmup chain
# - run_chain: integral chain
# - plot_dir: directory to save the plot

# and the following methods:
# - __init__: initialize the VMC
# - warmup: warmup the VMC
# - run: integrate the VMC
# - get_energy: get the energy
# - get_energy_std: get the energy standard deviation using blocking


class VariationalMonteCarlo:
    def __init__(self, system, walker, 
                 params=np.array([0.]),
                 warmup_steps=1000, run_steps=10000, 
                 block_quantiles=0.8,
                 plot=False, plot_dir=None):
        self.system = system
        self.walker = walker

        self.params = params

        self.warmup_steps = warmup_steps
        self.run_steps = run_steps

        self.block_quantiles = block_quantiles

        self.plot = plot
        self.save_plot = plot_dir

        self.warmup_chain = None
        self.run_chain = None
        self.run_analizer = None

        self.system.set_params(self.params)

    def set_params(self, params):   
        # set the variational parameters
        self.params = params
        self.system.set_params(self.params)

    def warmup(self):
        # warmup the VMC
        self.warmup_chain = self.walker.get_chain(self.warmup_steps)

        if self.plot:
            analizer = Analizer(self.system, self.warmup_chain)
            print('Acceptance rate:', analizer.get_acceptance_rate())
            plt.figure()
            plt.plot(analizer.get_local_energies())
            plt.xlabel('steps')
            plt.ylabel('loc energy')
            name = 'Warmup, params:'
            for p in self.params:
                name += '_'+('%.3f' % p)
            plt.title(name)
            plt.show()
            if self.save_plot is not None:
                # make a name that contains the params
                plt.savefig(self.save_plot+name+'.png')

    def run(self):
        # integrate the VMC
        self.run_chain = self.walker.get_chain(self.run_steps)
        self.run_analizer = Analizer(self.system, self.run_chain, self.block_quantiles)

        if self.plot:
            print('Acceptance rate:', self.run_analizer.get_acceptance_rate())
            plt.figure()
            plt.plot(self.run_analizer.get_local_energies())
            plt.xlabel('steps')
            plt.ylabel('loc energy')
            name = 'Run, params:'
            for p in self.params:
                name += '_'+('%.3f' % p)
            plt.title(name)
            plt.show()
            if self.save_plot is not None:
                # make a name that contains the params
                plt.savefig(self.save_plot+name+'.png')
            plt.figure()
            block_std = self.run_analizer.get_block_std()
            energy_std = np.sort(block_std)[int(len(block_std)*0.8)]
            plt.plot(self.run_analizer.get_block_std(), 'o')
            plt.xlabel('blocking')
            plt.ylabel('loc energy std')
            plt.axhline(y=energy_std, color='r', linestyle='--')
            name = 'Blocking, params:'
            for p in self.params:
                name += '_'+('%.3f' % p)
            plt.title(name)
            plt.show()
            if self.save_plot is not None:
                # make a name that contains the params
                plt.savefig(self.save_plot+name+'.png')

    def get_energy(self):
        # get the energy
        return self.run_analizer.get_mean_energy()

    def get_energy_std(self):
        return self.run_analizer.get_std_energy()