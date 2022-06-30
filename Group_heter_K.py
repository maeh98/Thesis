#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 19:04:39 2022

@author: manuel
Stochastic model corresponding to the ODE model. Si and Se are assumed to be in
a fast sub-system, hence their derivatives are assumed to be zero such that their
value is fixed. Se is not explicitely expressed in the production term due to re-expression (there is X_mean instead).
Analysis of the stochastic model leads to master equations, which
are simulated.
X_mean is approximated as sum of i * pi for all states i. 
pi is the probability to be in state i: pi(t) := P(Xt = i) for Xt the number of
synthetase molecules in a cell at time t.

Compared to Group_sensing, the parameters are not the same for all cells.
The cells are considered in 5 classes of equal size, each with a different K value.
Each class has its own simulation, but the all contribute to X_mean.
"""
# Import packages
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint, RK45

# %% 

def Prod(state, X_mean, rho_v, fitted_params):
    (c_sec, gamma_ex, A_syn, V_min, V_max, K, gamma_x, r, carr_k, N0, M, zeta) = fitted_params

    # Hill parameter
    n_h = 2
    S = (A_syn / (zeta * gamma_ex)) * (rho_v * X_mean + gamma_ex * state / c_sec)
    prod = zeta * (V_min + (V_max - V_min) * (S ** n_h) / ((S ** n_h) + K ** n_h))
    
    return prod

def find_rho_v(t, r, carr_k, N0):
    """Find dynamic parameter rho_v(t). It depends on the cell population."""
    # Explicit solution of the logistic equation. 
    N = (N0 * carr_k * np.exp(r * t)) / (N0 * (np.exp(r * t) - 1) + carr_k)
    
    # Convert concentration to absolute number: approximation of estimate by 
    # Kim et al. (2012)
    N_cells = (N) * 2e8    # N_cells / mL   # + 4e6
    V_cell = 3.6e-12       # mL # Fujimoto Plos
    V_tot = 1.0          # mL

    # Dynamic parameter
    rho_v = (N_cells * V_cell) / V_tot

    return rho_v

# Compute the R.H.S. of the master equations as a vector.
def Markov_ODE_system(p, t, *args):
    """Difference to Group_sensing.py.
     - X_mean is calculated over the 5 cell classes.
     - Production vector is calculated for each class.
     - Master equation RHS results are stored in the same vector, so indexes should only access the appropriate part. Compare the Group_sensing to check indexes.
    """
    (c_sec, gamma_ex, A_syn, V_min, V_max, K, gamma_x, r, carr_k, N0, M, zeta) = args

    # Maybe vector doesn' sum up to one
    p = p / np.sum(p)

    # Make a vector of indices
    M=int(M)
    i = np.arange(M)

    # Pre-calculate rho_v, X_mean, as well as the production function
    rho_v = find_rho_v(t, r, carr_k, N0)    
    
    # Get X_mean for five different cell classes. 
    X_mean = 0
    for j in range(5):
        X_mean += (1 / 5) * np.sum(i * p[j * M: (j + 1) * M])

    # Create (semi-arbitrarily) 5 different K values
    K_values = K * np.array([0.5, 0.9, 1, 1.5, 3])
    
    # Find the production value for these 5 different K_values
    # Find the RHS of the ODE system
    # Need to make calculations only within class. That's why for the complicated indexing.
    dpdt = np.zeros(p.shape)   
    for j, K_temp in enumerate(K_values):
        args = (c_sec, gamma_ex, A_syn, V_min, V_max, K_temp, gamma_x, r, carr_k, N0, M, zeta)
        prod_vec = np.array([Prod(state, X_mean, rho_v, args) for state in i])
        M = int(M)
        dpdt[j * M] = gamma_x * p[1 + j * M] - zeta * V_min * p[j * M]
        dpdt[1 + j * M: -1 + (j + 1) * M] = - (gamma_x * i[1: -1] + prod_vec[1: -1]) * p[1 + j * M: -1 + (j + 1) * M] + (gamma_x * i[2: ]) * p[2 + j * M: (j + 1) * M] + prod_vec[:-2] * p[j * M: -2 + (j + 1) * M]
        dpdt[(j + 1) * M - 1] = - (gamma_x * i[-1] + prod_vec[-1]) * p[(j + 1) * M - 1] + prod_vec[-2] * p[(j + 1) * M - 2]
        
    return dpdt


# %% 
def rungeKutta(p0, *args, dt = 0.005, t0 = 0, tf = 13):
    """Numericall solves the (Master) ODE system according to the Runge Kutta 4 (RK4) algorithm."""
    # Number of iterations.
    n = (int)((tf - t0)/dt)
    
    # Initialization
    p = p0
    t = t0
    k1 = dt * Markov_ODE_system(p, t, *args)
    k2 = dt * Markov_ODE_system(p + 0.5 * k1, t + 0.5 * dt, *args)
    k3 = dt * Markov_ODE_system(p + 0.5 * k2, t + 0.5 * dt, *args)
    k4 = dt * Markov_ODE_system(p + k3, t + dt, *args)
    KA = (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)

    # Save results each time step
    p_ = np.zeros((n + 1, p0.shape[0]))
    p_[0, :] = p
    
    # First update
    p = p + KA
    p_[1, :] = p

    for i in range(1, n + 1):
        # Intermediate steps
        k1 = dt * Markov_ODE_system(p, t, *args)
        k2 = dt * Markov_ODE_system(p + 0.5 * k1, t + 0.5 * dt, *args)
        k3 = dt * Markov_ODE_system(p + 0.5 * k2, t + 0.5 * dt, *args)
        k4 = dt * Markov_ODE_system(p + k3, t + dt, *args)

        KB = (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)

        # Update based on the previous two RK steps to dampen instabilities.
        p = p + (KA + KB) / 2
        
        # Set very low numbers to zero to avoid issues
        p = p * (p > 1e-16)
        # Not really mathematical, but resize to 1
        p = p / np.sum(p)

        # Save values
        p_[i, :] = p
        # Update next value of t
        t = t + dt
        KA = KB

    return p, p_

# %%
def stoch_and_ODE_RK4_3(fitted_params, N0=0.07683966, t0=0, tf = 13, dt=0.01, self_simulation=True):
    """Simulate Master equations.
    Note: zeta: Converting concentration to number of molecules within cell.
    # Molecules in a cell = total #molecules * cell_volume / total_volume, where
    total #molecules = concentration (mol/L) * Avogadro n. * total volume.
    The conversion factor is used by multiplying to concentration, to get number of
    molecules per cell.
    Note2: If the state is big enough, the production - degradation of the synthetase is negative, so if we chose some state M big enough we can ignore all master equations for states bigger than M.
    Note3: The initial condition is chosen to be zero everywhere except at the extremes in order to find eventually find out whether two nodes coexist.
    """
    (c_sec, gamma_ex, A_syn, V_min, V_max, K, gamma_x, r, carr_k) = fitted_params
    zeta = 3.6e1    # Wrong, should be 3.6e8
    M = int(1.4 * zeta * V_max / gamma_x)
    print(f' There are {M} ODEs.')
    args = tuple(np.append(fitted_params, [N0, M, zeta]))
    
    # Initial conditions.    
    p0 = np.zeros(int(M) * 5, )
    p0[::M] = 1  
    p0[M - 1:: M] = 0
    
    if self_simulation:
        p, p_=rungeKutta(p0, *args, tf=tf)
    # else:
    #     t = np.linspace(t0, tf, n)
    #     # solve ODE
    #     # p_ = odeint(Markov_ODE_system, p0, t, args)
        # p_ = RK45(Markov_ODE_system2, t0, p0, tf, 0.001,
        #           1e-06, False, None, np.array(args))
    #     # Making sure that the probabilities don't stop summing up to one due
    #     # to numerical instabilities.
    #     for i in range(p_.shape[0]):
    #         p_[i, :] = p_[i, :] / np.sum(p_[i, :])

    # Number of iterations.
    n = (int)((tf - t0)/dt)
    # Plot
    M=int(M)
    states = np.arange(M)
    times = [0, int(n/4), int(n/2), int(3*n/4), int(0.9 * n), int(n)]
    p_ = p_.reshape((p_.shape[0], 5, M))
    
    for j in np.arange(5):
        plt.plot(states, p_[times[0], j, :M], 'g', label='time 0', alpha=0.5)
        plt.plot(states, p_[times[1], j, :M], 'y', label=f'time {times[1] * dt}', alpha=0.5)
        plt.plot(states, p_[times[2], j, :M], 'r', label=f'time {times[2] * dt}', alpha=0.5)
        plt.plot(states, p_[times[3], j, :M], 'm', label=f'time {times[3] * dt}', alpha=0.5)
        plt.plot(states, p_[times[4], j, :M], 'k', label=f'time {times[4] * dt}')
        plt.plot(states, p_[-1, j, :M], 'b*', label=f'time {times[5] * dt}', alpha=0.5)
        plt.ylim([0, 0.025])
        plt.title('Probability distribution over time -- Group Sensing', fontweight='bold')
        plt.xlabel("states", fontweight='bold')
        plt.ylabel('probability', fontweight='bold')
        plt.legend()
        plt.show()
    
    p_mean = np.mean(p_, axis=1)
    plt.plot(states, p_mean[times[0], :M], 'g', label='time 0', alpha=0.5)
    plt.plot(states, p_mean[times[1], :M], 'y', label=f'time {times[1] * dt}', alpha=0.5)
    plt.plot(states, p_mean[times[2], :M], 'r', label=f'time {times[2] * dt}', alpha=0.5)
    plt.plot(states, p_mean[times[3], :M], 'm', label=f'time {times[3] * dt}', alpha=0.5)
    plt.plot(states, p_mean[times[4], :M], 'k', label=f'time {times[4] * dt}')
    plt.plot(states, p_mean[-1, :M], 'b*', label=f'time {times[5] * dt}', alpha=0.5)
    plt.ylim([0, 0.025])
    plt.title('Probability distribution over time -- Group Sensing', fontweight='bold')
    plt.xlabel("states", fontweight='bold')
    plt.ylabel('probability', fontweight='bold')
    plt.legend()
    plt.show()

    return p_

