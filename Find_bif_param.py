#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 22:36:16 2022

@author: manuel

This file is useful to visualize some steps of the bifurcation analysis. It uses sympy to give symbolic expression to follow the analysis.
Note: It's still the old model: need to fix (should take little time to do so)
!!!!!!!!!!!!!!!!!! Maybe not, wait for fresh mind.

Some break up into smaller functions could make the code more readable. But also would take very long.
!!!!!!!!!!!
"""
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, Eq, solve, lambdify, preorder_traversal, Float, symbols#, Derivative


# %% ############# FIND BI-STABILITY REGION (if it exists) ###################
def bifurcation_value(fitted_params, n=2):
    """Parameter n is the hill parameter.
    We have bistability when Se = NetProd(Si); i.e. the net production function NetProd(Si) intersects the line Se = const. three times in the positive domain. So we look for inflection points in NetProd(Si). If there are we can find the bi-stability area in Si, and we can specify values of Se for which we predict bi-stability to actually occur.
    Either case, keeping all other parameters fixed, we can find for which parameter K we get an inflection in NetProd(Si) and hence possibility for bi-stability.
    """
    (c_sec, gamma_ex, A_syn, V_min, V_max, K, gamma_x, r, carr_k) = fitted_params
    # rho_v found from carrying capacity, since we assume to be in equilibrium, 
    # including cell population equilibrium.
    N_cells = carr_k  * 2e8     #+ 4e6    #cells / mL
    V_cell = 3.6e-12       # mL
    V_tot = 1.0          # mL
    rho_v = (N_cells * V_cell) / V_tot

    # create "symbol" x. Will represent Si
    x = Symbol('x', real=True, positive=True)
    
    # Define (Net Production) function, make it readible and save plottable version.
    # Hill parameter to be defined
    NetProd = x - (A_syn / (gamma_x*c_sec)) * (V_min + (V_max - V_min) * (1 - (K ** n / (x ** n + K ** n))))
    for a in preorder_traversal(NetProd):
        if isinstance(a, Float):
            NetProd = NetProd.subs(a, format(a,'.2E'))
    print(f'Net production function is: {NetProd}')
    NetProd_plot = lambdify(x, NetProd)
    
    # Find derivative, make it readible and save plottable version.
    derivative_NetProd = NetProd.diff(x)
    for a in preorder_traversal(derivative_NetProd):
        if isinstance(a, Float):
            derivative_NetProd = derivative_NetProd.subs(a, format(a,'.2E'))
    print(f'Its derivative function is: {derivative_NetProd}')
    derivative_NetProd_plot = lambdify(x, derivative_NetProd)


    # %% Find the zeros (in positive space) of the derivative (If they exist, still need to define conditions). This is in order to find inflection points in NetProd(Si)
    find_zeros = Eq(derivative_NetProd, 0)
    zeros_der = solve(find_zeros, x)
    zeros_der = np.array(zeros_der)
    zeros_der = zeros_der[zeros_der > 0]

    if len(zeros_der) == 0:
        print("Bistability can not happen, as the RHS has no local minima/maxima")
    else:
        print(f'Due to {len(zeros_der)} (positive) local maxima/minima, there is bistability for S_e values within {[np.round(float(NetProd_plot(zeros_der[0])), 4), np.round(float(NetProd_plot(zeros_der[1])), 4)]}')
    

    # %% Plot Production function
    Si = np.linspace(0, 1, 1001)

    plt.figure()
    plt.title('RHS of NetProd(Si)')
    if len(zeros_der) != 0:
        plt.plot([zeros_der[0], zeros_der[1]], [NetProd.subs(x, zeros_der[0]), NetProd.subs(x, zeros_der[1])], 'r*')
    plt.plot(Si, NetProd_plot(Si), 'b-')
    plt.xlabel('Si')
    plt.ylabel('Net Production')
    plt.show()


    # %% Plot derivative
    plt.figure()
    plt.title('RHS: re-expression of the derivative of NetProd(Si)')
    plt.plot(Si, 1 - derivative_NetProd_plot(Si), 'b-')
    plt.xlabel('Si')
    plt.ylabel('Derivative')
    plt.show()


    # %% #################### BIFURCATION ANALYSIS for K #################
    # Figure out parameter conditions for bi-stability. K is variable, everything
    # else fixed according to parameter estimation.
    # z stands for K**n, y for S_i/x
    y, z = symbols('y z', real=True, positive=True)
    const = gamma_x * c_sec / (A_syn * (V_max - V_min)) 
    
    # Conditions for bi-stability follow from const = RHS having a (2) solutions.
    # RHS is an expression after re-arrangement of 0 = d/dSi NetProd(Si). Const is the LHS.
    # Hence, I check the maximum value of RHS as a function of K**n and henceforth see
    # what values of K I need s.t. const = RHS has a real and positive solution.
    RHS = (n * (z) * y ** (n-1)) / ((y ** n) + (z)) ** 2
    
    # Find the maximiser of RHS: an expression of y (Si) in terms of z (K) s.t. RHS is maximised.
    RHS_der = RHS.diff(y)
    print(f'RHS is {RHS}')
    print(f'RHS_der is {RHS_der}')
    find_zeros2 = Eq(RHS_der, 0)
    zeros_der2 = solve(find_zeros2, y)
    zeros_der2 = np.array(zeros_der2)
    print(f'Number of solutions is {len(zeros_der2)}')
    print()
    print(f'All sols: {solve(Eq(RHS_der, 0), y)}')
    
    if len(zeros_der2) > 1:
        print(f'There are {len(zeros_der2)} solutions for k: {zeros_der2}.')
        raise ValueError("zeros_der2 should have one entry only!")
    else:
        pass

    # Input that value of y back into the expression for RHS.
    # Now, find the first (i.e. bifurcation) value of z (K) such that const and RHS are the same.
    y_star = zeros_der2[0]     
    new_RHS = RHS.subs(y, y_star)
    print(new_RHS)
    K_star = solve(new_RHS - const, z)
    
    # Check all is good!
    if len(K_star) > 1:
        print(f'There are {len(K_star)} solutions for k_star: {K_star}.')
        raise ValueError("K_star should have one entry only!")
    K_star = np.power(float(K_star[0]), 1 / n)
    print(f'K_star is {np.round(K_star, 2)}, all K values less than that have bi-stability.')
    print(f'K is {np.round(K, 2)}')

    return K_star


# %% SYMPY Cheat sheet! To remind oneself of the workings of sympy.
# # create a "symbol" called x. Keep real to avoid complex values later on.
# x = Symbol('x', real=True)
# # x, y = = symbols('x y') 
 
# #Define function
# f = np.pi * x**2 - 1
 
# #passing values to the function
# f.subs(x, 2)
# # Other smarter way
# f1 = lambdify(x, f)
# print(f1(np.linspace(0,2,5)))

# # Compute derivative ...
# derivative_f = f.diff(x)
# # ... and visualize it nicely
# ex2 = derivative_f
# for a in preorder_traversal(derivative_f):
#     if isinstance(a, Float):
#         ex2 = ex2.subs(a, format(a,'.2E'))
# print(derivative_f)  # original
# print(ex2)  # rounded

# # Define equation & Solve it
# eq1 = Eq(f, 0)
# sol = solve(eq1, x)

