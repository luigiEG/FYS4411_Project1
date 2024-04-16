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

from module.QuantumSystem import SHOscillator


# the class MetropolisWalker have the following attributes:
# - system: the quantum system
# - X: the initial position (if None, gauss is used)
# - delta: the step size
# - subgroup_size: subgroup of particle to move

# and the following methods:
# - __init__: initialize the walker
# - step: make a step
# - propose: propose a move
# - test: test the move
# - get_chain: get a chain of steps

spec = [
    ('system', SHOscillator.class_type.instance_type),
    ('X', float64[:,:]),
    ('delta', float64),
    ('subgroup_size', int32),
]

@jitclass(spec)
class MetropolisWalker:
    def __init__(self, system, X=None, delta=None, subgroup_size=None):
        self.system = system

        if X is None:
            self.X = np.random.randn(self.system.N, self.system.D)
        else:
            self.X = X

        if subgroup_size is None and delta is None:
            self.subgroup_size = int(system.N/3)
            if self.subgroup_size == 0:
                self.subgroup_size = 1
            self.delta = 1.0/np.sqrt(self.subgroup_size)
        
        if subgroup_size is None and delta is not None:
            self.subgroup_size = int(system.N/3)
            if self.subgroup_size == 0:
                self.subgroup_size = 1
            self.delta = delta
        
        if subgroup_size is not None and delta is None:
            self.subgroup_size = subgroup_size
            self.delta = 1.0/np.sqrt(self.subgroup_size)
       

    def step(self):
        # make a step
        X_new = self.propose()  
        if self.test(X_new):
            self.X = X_new

    def propose(self):
        # propose a move
        # random sample of subgroup_size particles, no repetition
        subgroup_indx = np.random.choice(self.system.N, self.subgroup_size, replace=False)
        dX = np.zeros((self.system.N, self.system.D)) 
        dX[subgroup_indx] = (np.random.randn(self.subgroup_size,self.system.D))*self.delta
        return self.X + dX

    def test(self, X_new):
        # test the move
        likelihood = np.exp(2*(self.system.log_wavefunction(X_new) - self.system.log_wavefunction(self.X)))
        return np.random.random() < likelihood

    def get_chain(self, n_steps):
        # get a chain
        chain = np.zeros((n_steps, self.system.N, self.system.D))
        for i in range(n_steps):
            self.step()
            chain[i] = self.X
        return chain