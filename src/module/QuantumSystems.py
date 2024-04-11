import numpy as np
from math import exp, sqrt
from numba import int32, float64, types
from numba.experimental import jitclass


# the class QuantumSystem have the following attributes:
# - N: number of particles
# - D: number of dimensions
# - params: variational parameters (array)

# and the following methods:
# - __init__: initialize the system
# - set_params: set the variational parameters
# - wavefunction: compute the wavefunction
# - local_energy: compute the local energy

spec = [
    ('N', int32),
    ('D', int32),
    ('params', float64[:]),  # assuming params is a 1D array of floats
]

@jitclass(spec)
class SHOscillatorFree:
    def __init__(self, N, D, params=np.array([0.])):
        self.N = N
        self.D = D
        self.params = params

    def set_params(self, params):
        # set the variational parameters
        self.params = params

    def wavefunction(self, X):
        # compute the wavefunction
        return exp(-self.params[0]*np.sum(X**2))
    
    def local_energy(self, X):
        # compute the local energy
        return self.N*self.D*self.params[0] + (0.5 - 2*self.params[0]**2)*np.sum(X**2)
    