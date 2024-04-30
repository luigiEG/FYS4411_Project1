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
    ('a_HS', float64)
]

@jitclass(spec)
class SHOscillator:
    def __init__(self, N, D, params=np.array([0., 0.]), a_HS=0.01):
        self.N = N
        self.D = D
        self.params = params
        self.a_HS = a_HS

    def set_params(self, params):
        # set the variational parameters
        self.params = params

    def wavefunction(self, X):
        # compute the wavefunction
        return exp(-self.params[0]*np.sum(X**2))
    
    def log_wavefunction(self, X):
        # compute the log of the wavefunction
        dx = self.get_dx(X)
        corr_term = 0
        # if any dx < self.a_HS, return -inf
        dx_flatten = dx.flatten()
        dx_nozero = dx_flatten[dx_flatten != 0]
        if np.any(dx_nozero < self.a_HS):
            corr_term = -np.inf
        else:
            for k in range(self.N):
                for j in range(k+1, self.N):
                    corr_term += np.log(1-self.a_HS/dx[j,k])
        # pritn the two terms
        return -self.params[0]*np.sum(X**2) + corr_term
    


    
    def get_dX(self, X):
        dX = np.zeros((self.N, self.N, self.D))
        for i in range(self.N):
            for j in range(i+1, self.N):
                dX[i,j,:] = X[i,:] - X[j,:]
        
        dX = dX - dX.transpose(1,0,2)
        return dX
    
    def get_dx(self, X):
        dx = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1, self.N):
                dx[i,j] = np.linalg.norm(X[i,:] - X[j,:])
        
        dx = dx + dx.transpose()
        return dx
    
    def get_u_prime(self, dx):
        # exploiting the fact that dx is symmetric
        u_prime = np.zeros_like(dx)
        for i in range(self.N):
            for j in range(i+1, self.N):
                u_prime[i,j] = self.a_HS/(dx[i,j]**2 - self.a_HS*dx[i,j])
        
        u_prime = u_prime + u_prime.transpose()
        return u_prime
    
    def u_second(self, dx):
        return self.a_HS*(self.a_HS-2*dx)/(dx**2 - self.a_HS*dx)**2
    
    def get_grad_phi_over_phi(self, X):
        grad_phi = np.zeros_like(X)
        grad_phi[:,0:2] = X[:,0:2]
        # grad_phi[:,2] = self.params[1]*X[:,2]
        grad_phi[:,2] = X[:,2]
        grad_phi = -2*self.params[0]*grad_phi
    
        return grad_phi
    
    def get_kin_sec_term(self, X, dX, dx, u_prime, grad_phi_over_phi):
        sec_term = 0 
        for k in range(self.N):
            corr_term = np.zeros_like(X[k,:])
            for j in range(self.N):
                if j != k:
                    corr_term += dX[k,j]*u_prime[k,j]/dx[k,j]
            sec_term += 2 * np.dot(grad_phi_over_phi[k,:], corr_term)
        return sec_term
    
    def get_kin_third_term(self, dX, dx, u_prime):
        third_term = 0
        for k in range(self.N):
            for j in range(self.N):
                for i in range(self.N):
                    if i != k and j != k:
                        third_term += np.dot(dX[k,j], dX[k,i]) * u_prime[k,j] * u_prime[k,i] / (dx[k,j]*dx[k,i])
        return third_term
    
    def get_kin_fourth_term(self, dx, u_prime):
        fourth_term = 0
        for k in range(self.N):
            for j in range(self.N):
                if j != k:
                    fourth_term += self.u_second(dx[k,j]) + 2 * u_prime[k,j]/ dx[k,j]
        return fourth_term
    
        
    def local_energy(self, X):
        # compute the local energy
        free_term = self.N*self.D*self.params[0] + (0.5 - 2*self.params[0]**2)*np.sum(X**2)
        corr_term = 0

        dX = self.get_dX(X)
        dx = self.get_dx(X)
        u_prime = self.get_u_prime(dx)
        grad_phi_over_phi = self.get_grad_phi_over_phi(X)

        sec_term = self.get_kin_sec_term(X, dX, dx, u_prime, grad_phi_over_phi)
        third_term = self.get_kin_third_term(dX, dx, u_prime)
        fourth_term = self.get_kin_fourth_term(dx, u_prime)

        return free_term -0.5*(sec_term + third_term + fourth_term)


