import numpy as np
from math import exp, sqrt
from numba import int32, float64, types
from numba.experimental import jitclass
from numba import njit




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
# - get_chain: get the chain of positions

def MetropolisPropose(system, X, delta, subgroup_size):
    # propose a move
    subgroup_indx = np.random.randint(0,system.N,subgroup_size)
    dX = np.zeros((system.N, system.D)) 
    dX[subgroup_indx] = (np.random.random((subgroup_size,system.D))-0.5)*delta
    return X + dX

def MetropolisTest(system, X_new, X):
    # test the move
    likelihood = system.wavefunction(X_new)**2/system.wavefunction(X)**2
    return np.random.random() < likelihood

def MetropolisChain(system, X, delta, subgroup_size, n_steps):
    # get a chain
    chain = np.zeros((n_steps, system.N, system.D))
    for i in range(n_steps):
        X_new = MetropolisPropose(system, X, delta, subgroup_size)
        if MetropolisTest(system, X_new, X):
            X = X_new
        chain[i] = X
    return chain


class MetropolisWalker:
    def __init__(self, system, X=None, delta=0.1, subgroup_size=1):
        self.system = system
        self.delta = delta
        self.subgroup_size = subgroup_size
        if X is None:
            self.X = np.random.randn(self.system.N, self.system.D)
        else:
            self.X = X

    def get_chain(self, n_steps):
        return MetropolisChain(self.system, self.X, self.delta, self.subgroup_size, n_steps)