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
class SHOscillator:
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
    
    def loc_kinetic_energy_from_wf(self, X):
        # calculate the laplacian of the wavefunction numerically
        dx = 1e-4
        laplacian = 0
        for i in range(self.D):
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[i] += dx
            X_minus[i] -= dx
            laplacian += (self.wavefunction(X_plus) + self.wavefunction(X_minus) - 2*self.wavefunction(X))/dx**2

        return -0.5* laplacian/ self.wavefunction(X)
    
    def log_wavefunction(self, X):
        # compute the log of the wavefunction
        return -self.params[0]*np.sum(X**2)
    
    def loc_kinetic_energy_from_logwf(self, X):
        # get the gradient of the log_wavefunction, numerically
        grad = np.zeros((self.N, self.D))
        dx = 1e-4

        laplacian = 0
        # adjust dimension of X, add a dimension
        X = X.reshape(self.N, self.D)
        for i in range(self.N):
            for j in range(self.D):
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[i,j] += dx
                X_minus[i,j] -= dx
                log_wf_X = self.log_wavefunction(X)
                log_wf_X_plus = self.log_wavefunction(X_plus)
                log_wf_X_minus = self.log_wavefunction(X_minus)
                grad[i,j] = (log_wf_X_plus - log_wf_X_minus)/(2*dx)
                laplacian += (log_wf_X_plus + log_wf_X_minus - 2*log_wf_X)/dx**2
            
        # readjust X, remove the added dimension
        
        return -0.5 * (laplacian + np.sum(grad**2))

    
    def local_energy(self, X):
        # compute the local energy using the numerical laplacian
        return 0.5*np.sum(X**2) + self.loc_kinetic_energy_from_logwf(X) 
    



