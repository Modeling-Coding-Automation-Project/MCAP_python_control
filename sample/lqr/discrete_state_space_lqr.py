"""
File: discrete_state_space_lqr.py

This script demonstrates the design and simulation of a discrete-time Linear-Quadratic Regulator (LQR) for a pendulum system. It includes the following main functionalities:
- Definition of a continuous-time state-space model for a pendulum.
- Discretization of the continuous model.
- LQR gain calculation using the Arimoto-Potter method and comparison with other methods.
- Simulation of the closed-loop system with reference tracking.
- Visualization of the simulation results using a plotting utility.

Reference URL:
https://github.com/AtsushiSakai/PyAdvancedControl
https://jp.mathworks.com/help/control/ref/lti.lqr.html
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import control
import numpy as np
import scipy.linalg as la

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter

simulation_time = 10.0
dt = 0.1

# pendulum model continuous
Ac = np.matrix([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, -0.1, 3.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, -0.5, 30.0, 0.0]
])

Bc = np.matrix([
    [0.0],
    [2.0],
    [0.0],
    [5.0]
])

Cc = np.matrix([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
])

Dc = np.matrix([
    [0.0],
    [0.0]
])

# Discretize the continuous time model
sys_d = control.c2d(control.ss(Ac, Bc, Cc, Dc), dt, method='euler')
Ad = sys_d.A
Bd = sys_d.B
Cd = sys_d.C
Dd = sys_d.D

# LQR parameters
Q = np.diag([1.0, 0.0, 1.0, 0.0])
R = np.diag([1.0])


def process(x, u):
    x = Ad * x + Bd * u
    return (x)


def lqr_with_arimoto_potter(Ac, Bc, Q, R):
    n = len(Bc)

    # Hamiltonian
    Hamiltonian = np.vstack(
        (np.hstack((Ac, - Bc * la.inv(R) * Bc.T)),
         np.hstack((-Q, -Ac.T))))

    eigen_values, eigen_vectors = la.eig(Hamiltonian)

    V1 = None
    V2 = None

    minus_count = 0
    for i in range(2 * n):
        if eigen_values[i].real < 0:
            if V1 is None:
                V1 = eigen_vectors[0:n, i]
                V2 = eigen_vectors[n:2 * n, i]
            else:
                V1 = np.vstack((V1, eigen_vectors[0:n, i]))
                V2 = np.vstack((V2, eigen_vectors[n:2 * n, i]))

            minus_count += 1
            if minus_count == n:
                break

    V1 = np.matrix(V1.T)
    V2 = np.matrix(V2.T)

    P = (V2 * la.inv(V1)).real

    K = la.inv(R) * Bc.T * P

    return K


def dlqr_origin(Ad, Bd, Q, R):
    """
    This is the original dlqr function from Python Control Library
    """

    N = np.zeros((Q.shape[0], R.shape[1]))

    # Solve the discrete-time algebraic Riccati equation
    X = la.solve_discrete_are(Ad, Bd, Q, R, e=None, s=N)

    # Compute the LQR gain
    K = np.linalg.solve(Bd.T @ X @ Bd + R, Bd.T @ X @ Ad)

    return K


def main_reference_tracking():
    # design LQR controller
    K = lqr_with_arimoto_potter(Ac, Bc, Q, R)
    # K, _, _ = control.lqr(Ac, Bc, Q, R)
    # K = dlqr_origin(A, B, Q, R)
    # K, _, _ = control.dlqr(A, B, Q, R)

    print("K: ")
    print(K)

    # prepare simulation
    t = 0.0

    x = np.matrix([
        [0.0],
        [0.0],
        [0.0],
        [0.0]
    ])
    u = np.matrix([0])

    x_ref = np.matrix([
        [1.0],
        [0.0],
        [0.0],
        [0.0]
    ])

    time = []
    plotter = SimulationPlotter()

    # simulation
    while t <= simulation_time:
        u = K * (x_ref - x)
        u0 = float(u[0, 0])

        x = process(x, u0)

        plotter.append(x_ref)
        plotter.append(x)
        plotter.append(u)

        time.append(t)
        t += dt

    # plot
    plotter.assign("x", column=0, row=0, position=(
        0, 0), x_sequence=time, label="px")
    plotter.assign("x", column=2, row=0, position=(
        0, 0), x_sequence=time, label="theta")
    plotter.assign("x_ref", column=0, row=0, position=(0, 0),
                   x_sequence=time, line_style="--", label="px_ref")
    plotter.assign("x_ref", column=2, row=0, position=(0, 0),
                   x_sequence=time, line_style="--", label="theta_ref")

    plotter.assign("u", column=0, row=0, position=(
        1, 0), x_sequence=time, label="input_force")

    plotter.plot("LQR Tracking")


if __name__ == '__main__':
    main_reference_tracking()
