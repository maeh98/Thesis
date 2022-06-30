#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 18:52:45 2022

@author: manuel
"""

"""
This model estimates parameters for a logistic equation.
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import leastsq

# Define logistic equation.
def logistic(y, t, *args):
    r = args[0]             # Initial exponential growth rate
    k = args[1]             # Carrying capacity
    N = y[0]
     
    dN = r * N * (1 - N / k)
    return dN


def find_r_k(OD, exp_t, plots=True):
    # Choose!
    exp_N = np.array(OD)

    # Initial Condition
    N_0 = OD[0]

    # Parameter initialization
    initial_guess = [1, 2]

    # Timesteps
    n_steps = exp_t[-1] * 100 + 1
    t = np.linspace(0, exp_t[-1]  , int(n_steps))

    # Evaluate residuals between real data and simulation data.
    def residuals(p):
        p = tuple(p)
        sim_N = odeint(logistic, N_0, t, args = p).flatten()
        res = [sim_N[int(i)] for i in 100 * exp_t] - exp_N
        return res.flatten()

    # Fit parameters
    fitted_params = leastsq(residuals, initial_guess)[0]
    
    # Plot results
    if plots == True:
        plt.plot(exp_t, exp_N, 'ro')
        plt.plot(t, odeint(logistic, N_0, t, args = tuple(fitted_params)), 'b.')
        plt.legend(['Experimental Data', 'Simulated Data'], loc = 'best')
        plt.xlabel('Time')
        plt.ylabel('Product Concentration')
        plt.title("Cell population -- Logistic equation")
        plt.show()

    return fitted_params    


# print(np.round(fitted_params, 3))

# # Calculate statistical value of the model
# sim_N = odeint(michelis_menten, N_0, t, args = tuple(fitted_params)).flatten()
# res = residuals(fitted_params)
# SS_res = np.sum(res ** 2)
# SS_tot = np.sum((exp_N - np.mean(exp_N)) ** 2)
# R2 = 1 - (SS_res / SS_tot)
# print("R2 is: ", np.round(R2, 3))

# print("The initial exponential growth rate is approximately: ", np.round(fitted_params[0], 3))
# print("Carrying capacity is: ", np.round(fitted_params[1], 3))
# # [0.92274828 3.9630387 ]