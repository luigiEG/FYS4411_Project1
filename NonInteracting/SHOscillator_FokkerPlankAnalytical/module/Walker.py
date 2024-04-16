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
    ('QF', float64[:,:]),
    ('delta', float64),
    ('subgroup_size', int32),
]

@jitclass(spec)
class FokkerPlankWalker:
    def __init__(self, system, X=None, QF=None, delta=None, subgroup_size=None):
        self.system = system

        if X is None:
            self.X = np.random.randn(self.system.N, self.system.D)
        else:
            self.X = X

        self.QF = self.system.quantum_force(self.X)

        if (subgroup_size is None) and (delta is None):
            self.subgroup_size = self.system.N
            self.delta = 0.01
        
        if subgroup_size is None and delta is not None:
            self.subgroup_size = self.system.N
            self.delta = delta
        
        if subgroup_size is not None and delta is None:
            self.subgroup_size = subgroup_size
            self.delta = 1.0/np.sqrt(self.subgroup_size)
        
        if subgroup_size is not None and delta is not None:
            self.subgroup_size = subgroup_size
            self.delta = delta
       

    def step(self):
        # make a step
        X_new = self.propose()  
        QF_new = self.system.quantum_force(X_new)
        if self.test(X_new, QF_new):
            self.X = X_new
            self.QF = QF_new

    def propose(self):
        # propose a move
        # random sample of subgroup_size particles, no repetition
        subgroup_indx = np.random.choice(self.system.N, self.subgroup_size, replace=False)
        dX = np.zeros((self.system.N, self.system.D)) 
        dX[subgroup_indx] = (np.random.randn(self.subgroup_size,self.system.D))*sqrt(self.delta) + self.QF[subgroup_indx]*self.delta*0.5
        return self.X + dX

    def test(self, X_new, QF_new):
        # test the move
        a = 0.5 * ( self.QF + QF_new ) 
        b = 0.5 * self.delta * 0.5 * ( self.QF - QF_new ) + (self.X - X_new) 
        GreenLikehood = exp( np.dot(a.flatten(), b.flatten()) )
        WaveFuncLikehood = np.exp(2*(self.system.log_wavefunction(X_new) - self.system.log_wavefunction(self.X)))
        return np.random.random() < (WaveFuncLikehood * GreenLikehood )



    def get_chain(self, n_steps):
        # get a chain
        chain = np.zeros((n_steps, self.system.N, self.system.D))
        for i in range(n_steps):
            self.step()
            chain[i] = self.X
        return chain