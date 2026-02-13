"""
File: discrete_state_space_1.py

This script demonstrates the simulation and C++ code generation of a discrete-time state-space model using Python.
It defines a simple state-space system, simulates its response over a number of time steps, and visualizes the results.
Additionally, it generates C++ header files for the defined state-space model for deployment purposes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import control
import matplotlib.pyplot as plt

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter

# define state-space model
A = np.array([[0.7, 0.2],
              [-0.3, 0.8]])
B = np.array([[0.1],
              [0.2]])
C = np.array([[2, 0]])
D = np.array([[0]])

dt = 0.01
sys = control.ss(A, B, C, D, dt)

# initialize state and input
x = np.array([[0],
              [0]])
u = 1  # input
n_steps = 50  # number of steps

plotter = SimulationPlotter()

# simulation
for _ in range(n_steps):
    y = C @ x + D * u
    x = A @ x + B * u

    plotter.append(x)
    plotter.append(y)

plotter.assign("x", column=0, row=0, position=(0, 0))
plotter.assign("x", column=1, row=0, position=(0, 0))
plotter.assign("y", column=0, row=0, position=(1, 0))

plotter.plot("Discrete-Time State-Space Response")
