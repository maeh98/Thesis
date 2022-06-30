#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:49:45 2022

@author: manuel

Stochastic model corresponding to the ODE model. Analysis of the stochastic model leads to master equations. This file examines a stationary solution situation of the master equations and finds the distibution of p_i in such a scenario.
pi is the probability to be in state i: pi(t) := P(Xt = i) for Xt the number of
synthetase molecules in a cell at time t.
Si and Se are assumed equal.
????
"""

# %% Import packages
import numpy as np
import matplotlib.pyplot as plt

# zeta = 3.6e8
zeta = 3.6e1    # Arbitrary, to make to computable by python.
"""
zeta: Converting concentration to number of molecules within cell.
# Molecules in a cell = total #molecules * cell_volume / total_volume, where
total #molecules = concentration (mol/L) * Avogadro n. * total volume.
The conversion factor is used by multiplying to concentration, to get number of
molecules per cell. 

In practice the value is too big and it leads to python not being able to process
the calculation. So zeta is artificially fixed. 
?????
"""

def Prod_eqilibrium(state, fitted_params):
    (c_sec, gamma_ex, A_syn, V_min, V_max, K, gamma_x, r, carr_k) = fitted_params
    
    # Hill parameter --- fixed.
    n_h = 2

    # rho_v found from carrying capacity, since we assume to be in equilibrium, 
    # including cell population equilibrium.
    N_cells = carr_k  * 2e8     #+ 4e6    #cells / mL
    V_cell = 3.6e-12       # mL
    V_tot = 1.0          # mL
    rho_v = (N_cells * V_cell) / V_tot

    S = (A_syn / gamma_x) * (rho_v * state + gamma_x * state / c_sec)
    prod = zeta * (V_min + (V_max - V_min) * (S ** n_h) / ((S ** n_h) + (zeta * K) ** n_h))
    
    return prod


# %% Find equilibrium solution
def eq_pis(fitted_params, log_plot=False):
    (c_sec, gamma_ex, A_syn, V_min, V_max, K, gamma_x, r, carr_k) = fitted_params

    # Calculate how many "states" need to be tracked. Based on flow equilibria.
    M = int(2 * zeta * V_max / gamma_x)
    print("M is :", M)

    # Initialise
    all_pis = np.zeros(M)
    all_pis[0] = 1
    
    # All derivatives set to 0 (equilibrium) results in expression of p_i in
    # terms of p_(i-1), p_(i-2)
    for j in range(1, M):
        all_pis[j] = Prod_eqilibrium(j, fitted_params) * all_pis[j - 1] / (gamma_x * j)

    # Rescale s.t. sum p_i = 1
    all_pis_ = all_pis / np.sum(all_pis)
    print(np.sum(all_pis_ * np.arange(M)))

    # Plots
    plt.plot(np.arange(M), all_pis_)
    plt.title(f'Equilibrium probability distribution (estimated),\
              mean: {np.round(np.sum(all_pis_ * np.arange(M)))}', fontweight='bold')
    plt.ylim([0, 0.2])
    plt.xlabel("states", fontweight='bold')
    plt.ylabel('probability', fontweight='bold')
    plt.show()
    
    if log_plot:
        plt.plot(np.arange(1, M), all_pis_[1:])
        plt.xscale("log")
        plt.ylim([0, 0.2])
        plt.xlabel("states - Log scale", fontweight='bold')
        plt.ylabel('probability', fontweight='bold')
        plt.title(f'Equilibrium probability distribution (estimated),\
                  mean: {np.round(np.sum(all_pis_ * np.arange(M)))}', fontweight='bold')
        plt.show()
    

    return all_pis


