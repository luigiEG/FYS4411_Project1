import os
import numpy as np
from math import exp, sqrt
import random

import timeit
import time

from numba import njit
from numba.experimental import jitclass
from numba import int32, float64, types


import matplotlib.pyplot as plt

# the class Analizer have the following attributes:
# - system: the quantum system
# - chain: the chain of positions

# and the following methods:
# - __init__: initialize the analizer
# - get_local_energies: get the local energies
# - get_acceptance_rate: get the acceptance rate
# - block_transform: block transform the energies
# - get_block_std: get the block standard deviation

@njit
def get_acceptance_rate(local_energies):
    # get the local energies, make np.diff and count how many zeros
    return np.count_nonzero(np.diff(local_energies))/len(local_energies)

@njit
def get_local_energies(system, chain):
        # get the local energies
        return np.array([system.local_energy(X) for X in chain])

@njit
def get_bar_psi_over_psi(system, chain):
        # get the local energies
        return np.array([system.bar_psi_over_psi(X) for X in chain])

@njit
def block_transform(energies):
    energies_prime = np.zeros(len(energies)//2)
    for i in range(len(energies)//2):
        energies_prime[i] = 0.5*(energies[2*i] + energies[2*i+1])
    return energies_prime

@njit
def get_block_std(energies):
    energies_prime = energies
    block_std = np.zeros(int(np.log2(len(energies_prime))) + 1)
    block_std[0] = np.std(energies_prime)/sqrt(len(energies_prime) - 1)
    for i in range(len(block_std)-2):
        energies_prime = block_transform(energies_prime)
        block_std[i+1] = np.std(energies_prime)/sqrt(len(energies_prime) - 1)
    return block_std




class Analizer:
    def __init__(self, system, chain, block_quantiles=0.8):
        self.system = system
        self.chain = chain
        self.block_quantiles = block_quantiles

        self.local_energies = get_local_energies(self.system, self.chain)
        self.block_std = None

    def get_local_energies(self):
        return self.local_energies

    def get_acceptance_rate(self):
        return get_acceptance_rate(self.local_energies)
    
    def get_block_std(self):
        self.block_std = get_block_std(self.local_energies)
        return self.block_std
    
    def get_mean_energy(self):
        return np.mean(self.local_energies)
    
    def get_std_energy(self):
        self.get_block_std()
        return np.sort(self.block_std)[int(self.block_quantiles*len(self.block_std))]
    
    def get_bar_E(self):
        bar_psi_over_psi = get_bar_psi_over_psi(self.system, self.chain)
        first_term = np.mean(self.local_energies*bar_psi_over_psi)
        second_term = np.mean(bar_psi_over_psi)*np.mean(self.local_energies)
        return 2 * (first_term - second_term)
    









# the VMC class have the following attributes:
# - system: the quantum system
# - walker: the metropolis walker
# - params: the variational parameters

# - warmup_steps: number of warmup steps
# - run_steps: number of steps

# - calibrate_steps: number of calibrate steps
# - batch_steps: number of batch steps
# - acceptance_rate: the goal acceptance rate
# - factor: the factor to change the delta and subgroup_size

# - block_quantiles: the quantiles to compute the energy standard deviation

# - optimize_steps: number of optimization steps
# - eta: the learning rate
# - eta_decay: the learning rate decay

# - plot: if True plot the results 
# - warmup_chain: warmup chain
# - run_chain: integral chain
# - plot_dir: directory to save the plot
# - verbose: if True print the results

# and the following methods:
# - __init__: initialize the VMC
# - warmup: warmup the VMC
# - run: integrate the VMC
# - get_energy: get the energy
# - get_energy_std: get the energy standard deviation using blocking


class VMC:
    def __init__(self, system, walker, 
                 params=np.array([0.]),
                 warmup_steps=1000, run_steps=10000,
                 calibrate_steps=10000, batch_steps=1000, acceptance_rate=0.5, factor=0.8,
                 block_quantiles=0.8,
                 optimize_steps=10, eta=0.1, eta_decay=0.9,
                 plot=False, plot_dir=None, verbose=False):
        self.system = system
        self.walker = walker

        self.params = params

        self.warmup_steps = warmup_steps
        self.run_steps = run_steps

        self.calibrate_steps = calibrate_steps
        self.batch_steps = batch_steps
        self.acceptance_rate = acceptance_rate
        self.factor = factor

        self.block_quantiles = block_quantiles

        self.plot = plot
        self.save_plot = plot_dir
        self.verbose = verbose

        self.warmup_chain = None
        self.run_chain = None
        self.run_analizer = None
        self.run_time = None

        self.optimize_steps = optimize_steps
        self.eta = eta
        self.eta_decay = eta_decay

        self.system.set_params(self.params)

    def set_params(self, params):   
        # set the variational parameters
        self.params = params
        self.system.set_params(self.params)

    def warmup(self):
        # warmup the VMC
        self.warmup_chain = self.walker.get_chain(self.warmup_steps)

        if self.verbose:
            analizer = Analizer(self.system, self.warmup_chain)
            print('-----------------')
            print('VMC warmup')
            print('-----------------')
            print('Parameters:', self.params)
            print('Acceptance rate:', analizer.get_acceptance_rate())

        if self.plot or self.save_plot is not None:
            analizer = Analizer(self.system, self.warmup_chain)
            plt.figure()
            plt.plot(analizer.get_local_energies())
            plt.xlabel('steps')
            plt.ylabel('loc energy')
            name = 'Warmup, params:'
            for p in self.params:
                name += '_'+('%.3f' % p)
            plt.title(name)
            plt.grid()
            if self.save_plot is not None:
                # make a name that contains the params
                plt.savefig(self.save_plot+name+'.png')
            if self.plot:
                plt.show()
            if self.plot == False:
                plt.close()
        
    def run(self):
        # integrate the VMC
        start = time.time()
        self.run_chain = self.walker.get_chain(self.run_steps)
        self.run_analizer = Analizer(self.system, self.run_chain, self.block_quantiles)
        end = time.time()
        self.run_time = end-start

        if self.verbose:
            # print the parameters, acceptance rate and energy and std
            print('-----------------')
            print('VMC run')
            print('-----------------')
            print('Parameters:', self.params)
            print('Acceptance rate:', self.run_analizer.get_acceptance_rate())
            print('Time: ', self.run_time)
            print('Energy:', self.run_analizer.get_mean_energy(), '+/-', self.run_analizer.get_std_energy())

        if False:
            plt.figure()
            plt.plot(self.run_analizer.get_local_energies())
            plt.xlabel('steps')
            plt.ylabel('loc energy')
            name = 'Run, params:'
            for p in self.params:
                name += '_'+('%.3f' % p)
            plt.title(name)
            plt.grid()
            if self.save_plot is not None:
                # make a name that contains the params
                plt.savefig(self.save_plot+name+'.png')
            if self.plot:
                plt.show()
            if self.plot == False:
                plt.close()
                
        if self.plot or self.save_plot is not None:
            plt.figure()
            block_std = self.run_analizer.get_block_std()[:-1]
            energy_std = np.sort(block_std)[int(len(block_std)*self.block_quantiles)]
            plt.plot(block_std, 'o')
            plt.xlabel('blocking')
            plt.ylabel('loc energy std')
            plt.axhline(y=energy_std, color='r', linestyle='--')
            name = 'Blocking, params:'
            for p in self.params:
                name += '_'+('%.3f' % p)
            plt.title(name)
            plt.grid()
            if self.save_plot is not None:
                # make a name that contains the params
                plt.savefig(self.save_plot+name+'.png')
            if self.plot:
                plt.show()
            if self.plot == False:
                plt.close()

    def calibrate(self):
        # calibrate the VMC
        for batch in range(self.calibrate_steps//self.batch_steps):
            batch_chain = self.walker.get_chain(self.batch_steps)
            batch_analizer = Analizer(self.system, batch_chain)
            batch_acceptance_rate = batch_analizer.get_acceptance_rate()
            if random.random() < 0.5:
                if batch_acceptance_rate < self.acceptance_rate:
                    self.walker.delta *= self.factor
                if batch_acceptance_rate > self.acceptance_rate:
                    self.walker.delta *= 1/self.factor
            else:
                if batch_acceptance_rate < self.acceptance_rate:
                    self.walker.subgroup_size = int(self.walker.subgroup_size*self.factor)
                    if self.walker.subgroup_size < 1:
                        self.walker.subgroup_size = 1
                if batch_acceptance_rate > self.acceptance_rate:
                    self.walker.subgroup_size = int(self.walker.subgroup_size/self.factor+0.5)
                    if self.walker.subgroup_size > self.system.N:
                        self.walker.subgroup_size = self.system.N
                        
        if self.verbose:
            print('-----------------')
            print('VMC calibrate')
            print('-----------------')
            print('delta:', self.walker.delta)
            print('subgroup_size:', self.walker.subgroup_size)
            print('Acceptance rate:', batch_acceptance_rate)

    def optimize(self):
        # optimize the VMC
        print("-----------------")
        print('VMC optimize')
        print("-----------------")
        print('Initial parameters:', self.params)
        for i in range(self.optimize_steps):
            # integrate the VMC
            self.calibrate()
            self.warmup()
            self.run()
            # get the gradient, update the parameters and the learning rate
            grad = self.run_analizer.get_bar_E()
            self.set_params(self.params - self.eta*grad)
            self.eta *= self.eta_decay
            # print the energy and the gradient
            print("-----------------")
            print('Parameters:', self.params)
            print('Energy:', self.run_analizer.get_mean_energy(), '+/-', self.run_analizer.get_std_energy())
            print('Gradient:', grad)


    def get_energy(self):
        # get the energy
        return self.run_analizer.get_mean_energy()

    def get_energy_std(self):
        # get the energy standard deviation using blocking
        return self.run_analizer.get_std_energy()
    
    def get_run_time(self):
        # get the run time
        return self.run_time   