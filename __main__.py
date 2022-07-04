#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 18:44:48 2022

@author: manuel

To run all of the file as a whole or partially.
STRUCTURE:
 - Import Packages.
 - Set the data (from outside sources). 
 - Fit the parameters for logistic growth of the cell colture. Unused later.
 - Fit the parameters to the data relating extracellular autoinducer, intracellular
 autoinducer, synthetase, cell population in an ODE model. The parameters are saved and used in later functions.
 - Have a corresponding stochastic model. To analyze it master equations are derived
 and are simulated.
 There are three possible simulations. Group_sensing deals with group sensing, in which 
 extracellular autoinducer Se changes in time. 
 To account for parameter heterogeneity, in Group_heter_K five types of cell "classes" are counted, each with their own parameter K. For simplicities sake, these "classes" are and remain of equal size, interacting only through the release of autoinducer.
 Self_sensing accounts for self-sensing, in which extracelluar autoinducer is fixed (it needs to be stated as a parameter). Self_heter_K is self_sensing but with 5 different cell classes with different K values.
 - The bifurcation function checks if with the current parameters one is the 
 theoretical bistability range. If it is, it gives the (theoretical)
 range of extracellular autoinducer Se such that a bi-stable mode exists. In any 
 case, the bifurcation parameter K is given.
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt


# %% Data
# Define data: OD represents (number of) cells, GFP represents (number of) synthase molecules.

# Experimental data
# Double mutants
OD1 = [0.000218877, 0.000711869, 0.002315261, 0.007530087, 0.02066667, 0.08033333,
       0.2433333, 0.728, 1.829, 2.702667, 3.356667, 4.016, 4.150667, 4.777]
exp_t1 = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]) - 6
GFP1 = [159.2, 83.7, 642.8, 8814.2, 11643.1, 28789.8, 15302.5, 14585,
        12778.2, 15405.8, 19249.9, 25856.9, 25287.9, 24284.9]

OD2 = [0.000530579, 0.00175489, 0.005804302, 0.01433333, 0.05233333, 0.156,
       0.537, 1.299333, 2.279667, 2.712667, 3.272667, 3.774333, 3.773333, 4.436]
exp_t2 = np.array([6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5,
                   14.5, 15, 16, 17, 18, 19]) - 6.5
GFP2 = [187.3, 592.2, 7188.8, 10375.2, 21996.4, 9411.1, 13299.4, 10681.8,
        12738.5, 19838.1, 20807.9, 27575.1, 26085.9, 25049.6]

OD3 = [0.000989, 0.002369, 0.005674, 0.015333, 0.029667, 0.07,
       0.187333, 0.480667, 1.201333, 2.823333, 3.738]
exp_t3 = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18]) - 6
GFP3 = [137.7, 167.9, 1628.8, 9062, 12188.8, 16691.7,
        30030.5, 12972.5, 12677.5, 22439.1, 25390.5]

# Wild Types
OD_WT1 = [0.006997, 0.014333, 0.027333, 0.067, 0.132, 0.187667, 0.234,
          0.324333, 0.649667, 1.018667, 1.462, 2.003333, 2.666667,2.836,
          3.062, 3.562667, 4.126, 4.135333]
exp_t_WT1 = np.array([8.5, 9.25, 10, 10.75, 11.5, 12, 12.5, 13, 13.5, 14,
                  14.5, 15, 15.5, 16, 17, 18, 19, 20]) - 8.5
GFP_WT1 = [49.2, 55.4, 51.8, 66.2, 61.3, 50, 48.6, 39.7, 45.6, 57.2, 735.2,
           5878.9, 12057.4, 15798.1, 21496.9, 20867.6, 19374.1, 18757.7]

OD_WT2 = [0.2556667, 0.6803333, 0.972, 1.548333, 2.081667, 2.778667, 3.226,
          3.157333, 4.111667, 4.011334]
exp_t_WT2 = np.array([12, 13, 13.5, 14, 14.5, 15, 15.5, 16, 17, 18]) - 12
GFP_WT2 = [47.6, 60.4, 76.5, 935.3, 4263.3, 7919.6, 11266.1,
           15315.7, 23118.9, 17549.1]

OD_WT3 = [0.2325, 0.4813334, 0.7073333, 1.012, 1.727333, 2.105667, 2.592667,
          2.871333, 3.463, 3.661333]
exp_t_WT3 = np.array([12, 13, 13.5, 14, 14.5, 15, 15.5, 16, 17, 18]) - 12
GFP_WT3 = [60.8, 56, 64.1, 124.9, 1299.3, 5089.6, 7708.8, 19401.3, 32376.9, 28927.9]


# %% Choose which data to use
OD = np.array(OD1)   # Value in 100 millions (it should be *1e8)
GFP = np.array(GFP1) * 1e-4    # Smaller than that and the fitting is very wierd
exp_t = exp_t1    # experimetal

if False:        # Just simple plot of the data.
    plt.figure()
    plt.plot(exp_t, OD, 'g*', linewidth=2)
    plt.xlabel("time")
    plt.ylabel("OD")
    plt.title("Cell concentration")
    plt.show()
    plt.figure()
    plt.plot(exp_t, GFP, 'b*', linewidth=2)
    plt.xlabel("time")
    plt.ylabel("GFP")
    plt.title("Sythetase concentration")
    plt.show()


# %% Fit the initial exponential growth rate of the population and the carrying capacity.
from Find_r_k_logistic import find_r_k
[r, k] = find_r_k(OD, exp_t, plots=False)
print(f'r is: {r :.4}, k is: {k :.4}')


# %% Find other model parameters through fitting to real data.
# rho_v dynamic parameter (function of population).
from Find_params import find_params
fitted_params = find_params(GFP, OD, exp_t, r, k, plots=True)
(c_sec, gamma_ex, A_syn, V_min, V_max, K, gamma_x, r, carr_k) = fitted_params
print(f'V_max/V_min = {V_max / V_min :.2e}')


# %%
"""
New model: stochastic rather than deterministic. Analysis give master equations.
This function outputs an estimate of the stationary solution of the master equations."""
from Finding_fixed_master import eq_pis
all_pis = eq_pis(fitted_params)


# %%
"""
Stochastic model corresponding to the ODE model. Si and Se are assumed to be in a fast sub-system, hence their derivatives are assumed to be zero such that their value is fixed. Analysis of the stochastic model leads to master equations, which are simulated.
X_mean is approximated as sum of i * pi
"""
import time
from Group_sensing import stoch_and_ODE_RK4
t1 = time.time()
p_ = stoch_and_ODE_RK4(fitted_params, N0=OD[0], t0=0, tf=13,
                       dt=0.03, self_simulation=True)
t2 = time.time()

print(f'Time taken: {(t2-t1): .3e}s')


# %% Find bifurcation value for K
"""This function is useful to visualize some steps of the bifurcation analysis. It uses sympy to give symbolic expression to follow the analysis. It is also useful to derive bifurcation parameter K_star.
"""
from Find_bif_param import bifurcation_value
K_star = bifurcation_value(fitted_params)


# %% 
"""Compared to Group_sensing, the parameters are not the same for all cells.
The cells are considered in 5 classes of equal size, each with a different K value.
Each class has its own simulation, but the all contribute to X_mean.
"""
import time
from Group_heter_K import stoch_and_ODE_RK4_3
t1 = time.time()
p_ = stoch_and_ODE_RK4_3(fitted_params, N0=OD[0], t0=0, tf = 15,
                         dt=0.005, self_simulation=True)
t2 = time.time()

print(f'Time taken: {(t2-t1): .3e}s')


# %%
"""Se is constant as for the self-sensing case we are not interested in externally-driven changes in Se concentration.
First entry of function is Se: to set it look 2 cells above at the bifurcation "domain" for Se. I Get the two bumps
"""
import time
from Self_sensing import stoch_and_ODE_RK4_2
t1 = time.time()
# p_ = stoch_and_ODE_RK4_2(0.00001, fitted_params_new, N0=0.07683966, t0=0, tf = 103, dt=0.01, self_simulation=True)
p_ = stoch_and_ODE_RK4_2(0.27, fitted_params, N0=OD[0], t0=0, tf = 100,
                         dt=0.01, self_simulation=True)
t2 = time.time()

print(f'Time taken: {(t2-t1): .3e}s')


# %%
"""
Compared to Self_sensing, the parameters are not the same for all cells.
The cells are considered in 5 classes of equal size, each with a different K value.
Each class has its own simulation, the classes are not interacting.
"""
import time
from Self_heter_K import stoch_and_ODE_RK4_2
t1 = time.time()
stoch_and_ODE_RK4_2(0.27, fitted_params, N0=0.07683966, t0=0, tf = 30, dt=0.01, self_simulation=True)
t2 = time.time()

print(f'Time taken: {(t2-t1): .3e}s')

