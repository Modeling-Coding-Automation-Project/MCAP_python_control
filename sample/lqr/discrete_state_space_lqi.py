"""
File Name: discrete_state_space_lqi.py

This script demonstrates the implementation of a Linear-Quadratic Integral (LQI) controller for a discrete-time state-space model of a pendulum system. The code sets up the continuous-time state-space representation, discretizes it, augments the system for integral action, and computes the optimal feedback gain using the Arimoto-Potter method. The simulation tracks a reference trajectory and visualizes the results using a custom plotting utility.

Reference URL:
https://github.com/AtsushiSakai/PyAdvancedControl
https://jp.mathworks.com/help/control/ref/lti.lqr.html
https://jp.mathworks.com/help/control/ref/ss.lqi.html
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

# Create Expanded State Space Model
Ac_ex = np.block([
    [Ac, np.zeros((Ac.shape[0], Cc.shape[0]))],
    [Cc, np.zeros((Cc.shape[0], Cc.shape[0]))]])
Bc_ex = np.vstack([Bc, np.zeros((Cc.shape[0], Bc.shape[1]))])

# Discretize the continuous time model
sys_d = control.c2d(control.ss(Ac, Bc, Cc, Dc), dt, method='euler')
Ad = sys_d.A
Bd = sys_d.B
Cd = sys_d.C
Dd = sys_d.D

# LQI parameters
Q_ex = np.diag([1.0, 0.1, 1.0, 0.1, 2.0, 0.1])
R_ex = np.diag([1.0])


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

    result = Hamiltonian @ eigen_vectors - \
        eigen_vectors @ np.diag(eigen_values)

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


def main_reference_tracking():

    K_ex = lqr_with_arimoto_potter(Ac_ex, Bc_ex, Q_ex, R_ex)
    # K_ex = control.lqr(Ac_ex, Bc_ex, Q_ex, R_ex)[0]

    print("K_ex: ")
    print(K_ex)

    K_x = K_ex[:, 0:Ac.shape[0]]
    K_e = K_ex[:, Ac.shape[0]:(Ac.shape[0] + Cc.shape[0])]

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

    y_ref = np.matrix([
        [1.0],
        [0.0]
    ])

    e_y_integral = np.zeros((2, 1))

    time = []
    plotter = SimulationPlotter()

    u_offset = 0.1

    # simulation
    while t <= simulation_time:
        y = Cc * x
        e_y = y_ref - y
        e_y_integral = e_y_integral + dt * e_y

        u = K_x * (x_ref - x) + K_e * e_y_integral
        u0 = float(u[0, 0]) + u_offset

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

    plotter.plot("LQI Tracking")


if __name__ == '__main__':
    main_reference_tracking()
