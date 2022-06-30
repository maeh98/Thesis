#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 20:34:30 2022

@author: manuel


This file simulates the model: X is stochastic and Si&Se are still "fixed", their derivatives are 0.

Stochastic model corresponding to the ODE model. Si is assumed to be in
a fast sub-system, hence its derivative is assumed to be zero such that Si's'
value is fixed. Se is constant as for the self-sensing case we are not interested in externally-driven changes in Se concentration.
Analysis of the stochastic model leads to master equations, which
are simulated.
pi is the probability to be in state i: pi(t) := P(Xt = i) for Xt the number of
synthetase molecules in a cell at time t.
"""
# Import packages
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint, RK45


# %% 

def Prod2(state, rho_v, fitted_params):
    (c_sec, gamma_ex, A_syn, V_min, V_max, K, gamma_x, r, carr_k, N0, M, zeta, Se) = fitted_params
    
    # Hill parameter
    n_h = 2
    S = Se + A_syn / (zeta * c_sec) * state
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
def Markov_ODE_system(p, t, *args):  # , *args
    (c_sec, gamma_ex, A_syn, V_min, V_max, K, gamma_x, r, carr_k, N0, M, zeta, Se) = args

    # Maybe vector doesn' sum up to one
    p = p / np.sum(p)

    # Make a vector of states
    states = np.arange(M)

    # Pre-calculate rho_v, X_mean, as well as the production function
    rho_v = find_rho_v(t, r, carr_k, N0)
    prod_vec = np.array([Prod2(j, rho_v, args) for j in states])

    # Find the RHS of the ODE system
    dpdt = np.zeros(p.shape)   
    dpdt[0] = gamma_x * p[1] - zeta * V_min * p[0]
    dpdt[1: -1] = - (gamma_x * states[1: -1] + prod_vec[1: -1]) * p[1: -1] + (gamma_x * states[2: ]) * p[2: ] + prod_vec[:-2] * p[: -2]
    dpdt[-1] = - (gamma_x * states[-1] + prod_vec[-1]) * p[-1] + prod_vec[-2] * p[-2]

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
def stoch_and_ODE_RK4_2(Se, fitted_params, N0=0.07683966, t0=0, tf = 13, dt=0.01, self_simulation=True, log_plot=False):
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
    args = tuple(np.append(fitted_params, [N0, M, zeta, Se]))
    
    # Initial conditions.    
    p0 = np.zeros(int(M), )
    p0[0] = 1
    p0[-1] = 0
    
    if self_simulation:
        p, p_=rungeKutta(p0, *args, tf=tf)
    # else:
    #     t = np.linspace(t0, tf, n)
    #     # solve ODE
    #     # p_ = odeint(Markov_ODE_system, p0, t, args)
    #     p_ = RK45(Markov_ODE_system2, t0, p0, tf, 0.001, 1e-06, False, None, np.array(args))
    #     # Making sure that the probabilities don't stop summing up to one due
    #     # to numerical instabilities.
    #     for i in range(p_.shape[0]):
    #         p_[i, :] = p_[i, :] / np.sum(p_[i, :])

    
    
    # Number of iterations.
    n = (int)((tf - t0)/dt)
    # Plot
    states = np.arange(M)
    times = [0, 10, 100, 1000, int(n/2), int(n-1)]
    plt.plot(states, p_[times[0], :], 'g', label='time 0')
    plt.plot(states, p_[times[1], :], 'y', label=f'time {times[1] * dt}')
    plt.plot(states, p_[times[2], :], 'r', label=f'time {times[2] * dt}')
    plt.plot(states, p_[times[3], :], 'm', label=f'time {times[3] * dt}')
    plt.plot(states, p_[times[4], :], 'k', label=f'time {times[4] * dt}')
    plt.plot(states, p_[-1, :], 'b*', label=f'time {times[5] * dt}')
    plt.ylim([0, 0.2])
    plt.title('Probability distribution over time -- Self sensing', fontweight='bold')
    plt.xlabel("states", fontweight='bold')
    plt.ylabel('probability', fontweight='bold')
    plt.legend()
    plt.show()

    print(f'Mean is : {np.round(np.sum(p_[-1, :] * np.arange(M)))}')
    if log_plot:
        plt.plot(states[1:], p_[times[0], 1:], 'g', label='time 0')
        plt.plot(states[1:], p_[times[1], 1:], 'y', label=f'time {times[1] * dt}')
        plt.plot(states[1:], p_[times[2], 1:], 'r', label=f'time {times[2] * dt}')
        plt.plot(states[1:], p_[times[3], 1:], 'm', label=f'time {times[3] * dt}')
        plt.plot(states[1:], p_[times[4], 1:], 'k', label=f'time {times[4] * dt}')
        plt.plot(states[1:], p_[-1, 1:], 'b*', label=f'time {times[5] * dt}')
        plt.ylim([0, 0.2])
        plt.title('Probability distribution over time -- Self sensing', fontweight='bold')
        plt.xlabel("states - Log scale", fontweight='bold')
        plt.ylabel('probability', fontweight='bold')
        plt.legend()
        plt.xscale("log")
        plt.show()

    return p_



# %%
# def Markov_ODE_system2(p, t, args):  # , *args
#     [c_sec, gamma_ex, A_syn, V_min, V_max, K, gamma_x, r, carr_k, N0, M, zeta] = args

#     # Maybe vector doesn' sum up to one
#     p = p / np.sum(p)

#     # Make a vector of indices
#     i = np.arange(M)

#     # Pre-calculate rho_v, X_mean, as well as the production function
#     rho_v = find_rho_v(t, r, carr_k, N0)
#     X_mean = np.sum(i * p)
#     prod_vec = np.array([Prod2(j, X_mean, rho_v, tuple(args)) for j in i])

#     # Find the RHS of the ODE system
#     dpdt = np.zeros(p.shape)   
#     dpdt[0] = gamma_x * p[1] - zeta * V_min * p[0]
#     dpdt[1: -1] = - (gamma_x * i[1: -1] + prod_vec[1: -1]) * p[1: -1] + (gamma_x * i[2: ]) * p[2: ] + prod_vec[:-2] * p[: -2]
#     dpdt[-1] = - (gamma_x * i[-1] + prod_vec[-1]) * p[-1] + prod_vec[-2] * p[-2]
#     # print(t, np.sum(dpdt), dpdt[-2])
#     return dpdt
