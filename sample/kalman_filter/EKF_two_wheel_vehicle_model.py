import os
import sys
sys.path.append(os.getcwd())

import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols
from dataclasses import dataclass

from python_control.kalman_filter import ExtendedKalmanFilter
from python_control.control_deploy import ExpressionDeploy

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter

# define parameters and variables
m, u, v, r, F_f, F_r = sp.symbols('m u v r F_f F_r', real=True)
I, l_f, l_r, v_dot, r_dot, V, beta, beta_dot = sp.symbols(
    'I l_f l_r v_dot r_dot V beta beta_dot', real=True)

eq_1 = sp.Eq(m * (v_dot + u * r), F_f + F_r)
eq_2 = sp.Eq(I * r_dot, l_f * F_f - l_r * F_r)

lhs = eq_1.lhs.subs({u: V, v_dot: V * beta_dot})
eq1 = sp.Eq(lhs, eq_1.rhs)

K_f, K_r, delta, beta_f, beta_r = sp.symbols(
    'K_f K_r delta beta_f beta_r', real=True)

# eq_vec = [eq_1.subs({F_f: -2 * K_f * beta_f, F_r: -2 * K_r * beta_r}),
#           eq_2.subs({F_f: -2 * K_f * beta_f, F_r: -2 * K_r * beta_r})]
rhs = eq1.rhs.subs({F_f: -2 * K_f * beta_f, F_r: -2 * K_r * beta_r})
eq_1 = sp.Eq(eq1.lhs, rhs)

rhs = eq_2.rhs.subs({F_f: -2 * K_f * beta_f, F_r: -2 * K_r * beta_r})
eq_2 = sp.Eq(eq_2.lhs, rhs)

# eq_vec = [eq.subs({beta_f: beta + (l_f / V) * r - delta,
#                   beta_r: beta - (l_r / V) * r}) for eq in eq_vec]
rhs = eq_1.rhs.subs({
    beta_f: beta + (l_f / V) * r - delta,
    beta_r: beta - (l_r / V) * r
})
eq_1 = sp.Eq(eq_1.lhs, rhs)

rhs = eq_2.rhs.subs({
    beta_f: beta + (l_f / V) * r - delta,
    beta_r: beta - (l_r / V) * r
})
eq_2 = sp.Eq(eq_2.lhs, rhs)

eq_vec = [eq_1, eq_2]

solution = sp.solve(eq_vec, beta_dot, dict=True)
beta_dot_sol = sp.simplify(solution[0][beta_dot])
print("beta_dot =\n", beta_dot_sol)

solution = sp.solve(eq_vec, r_dot, dict=True)
r_dot_sol = sp.simplify(solution[0][r_dot])
print("r_dot =\n", r_dot_sol)
