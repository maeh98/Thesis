#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 19:56:22 2022

@author: manuel

This file takes the first model in Fujimoto (and qStoch), and fits the model parameters to the data provided by Dr. Schuster. 

Parameter rho_v is treated as a dynamic parameter, and makes it necessary to include an additional dimension to the model: cell population (N). All other parameters are static.
"""


# %% Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import leastsq


# %% Paramaters from literature. Useful for initialization.
# rho_v = 0.01   # [N * V_cell / V_tot]
c_sec = 12    # [1 / h]
gamma_ex = 0.0174  # [1 / h]
A_syn = 1.6     # [1 / h]
V_min = 0.22    # [micro M]
V_max = 1.9     # [micro M]
K = 0.5         # [micro M / L]
gamma_x = 0.36  # [1 / h]
const = 1  # for now
r = 0.9643738
carr_k = 4.00056362


# %% Define ODE model
def RHS_full_fuji3(z, t, *args):
    """This function computes the right hand side of the ODE
    (starting point, full model Fujimoto).
    The 'unnecessary squares' are to enforce positivity of parameters in a simple way."""

    # Fixed parameter
    n_h = 2    # Hill parameter

    # Parameters to be fitted
    c_sec = args[0] 
    gamma_ex = args[1] 
    A_syn = args[2] 
    V_min = args[3] 
    V_max = args[4] 
    K = args[5] 
    gamma_x = args[6]
    r = args[7] 
    carr_k = args[8] 

    # Variables
    Se = z[0]
    Si = z[1]
    S_mean = z[1]
    X = z[2]
    N = z[3]
    
    # N_cells is number of cells, rather than their concentration
    # N_cells = ((N - 0.0764) / 2.0087) * 2e8 + 4e6    # N_cells / mL
    # Convert concentration to absolute number: approximation of estimate by Kim et al. (2012)
    N_cells = N * 2e8 #+ 4e6    # N_cells / mL
    V_cell = 3.6e-12       # mL # Fujimoto Plos
    V_tot = 1.0          # mL
    # Dynamic parameter
    rho_v = (N_cells * V_cell) / V_tot

    # ODE model as in Fujimoto Plos
    dSe = rho_v * (c_sec ** 2) * (S_mean - Se) - (gamma_ex ** 2) * Se
    dSi = (c_sec ** 2) * (Se - Si) + (A_syn ** 2) * X
    dX = (V_min ** 2) + ((V_max ** 2) - (V_min ** 2)) * (Si ** n_h) / (Si ** n_h + (K ** (2 * n_h))) - (gamma_x ** 2) * X
    dN = (r ** 2) * N * (1 - N / (carr_k ** 2))

    # Assemble the right hand side of the ODE system
    dzdt = [dSe, dSi, dX, dN]

    return dzdt

def find_params(GFP, OD, exp_t, r, k, plots=False):
    # Data
    exp_z = np.array([GFP, OD]).T
    
    # Initial condition
    IC = GFP[0]
    z0 = [IC, IC, IC, OD[0]]
    
    # Initialize parameters
    mm_params = tuple(np.sqrt((c_sec, gamma_ex, A_syn, V_min,
                               V_max, K, gamma_x, r, carr_k)))
    initial_guess = mm_params

    # Define Timesteps
    end = exp_t[-1]
    n_steps = end * 100 + 1
    t = np.linspace(0, end, int(n_steps))

    # Evaluate residuals between real data and simulation data.
    def residuals(p):
        p = tuple(p)
        sim_z = odeint(RHS_full_fuji3, z0, t, args=p)[:, 2:]
        res = np.array([sim_z[int(i)] for i in 100 * exp_t]).flatten() - exp_z.flatten()
        return res
    
    # Fit parameters
    fitted_params = leastsq(residuals, initial_guess)[0]

    # Plot results
    if plots == True:
        plt.plot(exp_t, exp_z[:, 0], 'ro')
        plt.plot(t, odeint(RHS_full_fuji3, z0, t, args = tuple(fitted_params))[:, 2], 'g.')
        plt.legend(['Experimental Data, GFP', 'X'], loc = 'best')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title("Synthetase dynamics")
        plt.show()

    if plots == True:
        plt.plot(exp_t, exp_z[:, 1], 'ro')
        plt.plot(t, odeint(RHS_full_fuji3, z0, t, args = tuple(fitted_params))[:, 3], 'k.')
        plt.legend(['Experimental Data, OD', 'N'], loc = 'best')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title("Cell population dynamics")
        plt.show()
    
    print(f'Fitted parameters (c_sec, gamma_ex, A_syn, V_min, V_max, K, gamma_x, r, carr_k) are: {fitted_params ** 2}')
    return (fitted_params ** 2)

# # Calculate statistical value of the model
# sim_N = odeint(RHS_full_fuji_first2, z0, t, args = tuple(fitted_params))[:, 2]
# res = residuals(fitted_params)
# SS_res = np.sum(res ** 2)
# SS_tot = np.sum((exp_N - np.mean(exp_N)) ** 2)
# R2 = 1 - (SS_res / SS_tot)
# print("R2 is: ", R2)
