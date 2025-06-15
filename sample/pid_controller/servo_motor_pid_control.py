"""
File: servo_motor_pid_control.py

This module provides functions for modeling the state-space representation of a DC servo motor with PID control.
It defines the state-space matrices (A, B, C, D) for the servo motor system, which are essential for control system analysis and design.
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from dataclasses import dataclass

from python_control.pid_controller import DiscretePID_Controller

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter


@dataclass
class ServoParams:
    R: float = 2.0           # Omega
    L: float = 5.0e-3        # H
    J: float = 1.0e-2        # kg·m^2
    B: float = 1.0e-3        # N·m·s/rad
    Kt: float = 0.1          # N·m/A
    Kb: float = 0.1          # V·s/rad


def servo_state_matrices(p: ServoParams):

    A = np.array([
        [0, 1, 0],
        [0, -p.B / p.J, p.Kt / p.J],
        [0, -p.Kb / p.L, -p.R / p.L]
    ])
    B = np.array([[0],
                  [0],
                  [1 / p.L]])
    C = np.array([
        [1, 0, 0],   # position θ
        [0, 1, 0]    # speed w
    ])
    D = np.zeros((2, 1))
    return A, B, C, D


if __name__ == "__main__":
    params = ServoParams()
    A, B, C, D = servo_state_matrices(params)

    plotter = SimulationPlotter()

    # euler forward method simulation
    sim_delta_time = 1e-3
    x = np.zeros((3, 1))
    u = np.array([[0.0]])
    simulation_time = 10.0

    # PID controller
    Kp = 2.0
    Ki = 0.5
    Kd = 1.0
    pid = DiscretePID_Controller(delta_time=sim_delta_time, Kp=Kp, Ki=Ki, Kd=Kd, N=(
        0.1 / sim_delta_time), Kb=Ki, minimum_output=-12.0, maximum_output=12.0)

    theta_ref = np.array([[1.0]])

    time = np.arange(0, simulation_time, sim_delta_time)
    np.random.seed(0)  # for reproducibility

    for k in range(round(simulation_time / sim_delta_time)):
        # plant response
        u = np.array(u).reshape((1, 1))
        u_offset = np.array([[0.1]])

        x_dot = A @ x + B @ (u + u_offset)
        x += sim_delta_time * x_dot           # x(k+1) = x(k) + dt * ẋ

        system_noise = np.random.normal(0, 0.01, size=(C.shape[0], 1))

        y = C @ x + D @ u + system_noise

        # PID controller
        e = theta_ref - y[0, 0]
        u = pid.update(e)

        plotter.append_name(theta_ref, "theta_ref")
        plotter.append_name(y, "y")

        u_max = pid.maximum_output
        u_min = pid.minimum_output
        plotter.append_name(u, "u")
        plotter.append_name(u_max, "u_max")
        plotter.append_name(u_min, "u_min")

        plotter.append_name(u_offset, "u_offset")
        plotter.append_name(system_noise, "system_noise")

    plotter.assign("theta_ref", column=0, row=0, position=(0, 0),
                   x_sequence=time, label="reference θ")
    plotter.assign("y", column=0, row=0, position=(0, 0),
                   x_sequence=time, label="position θ")
    plotter.assign("y", column=1, row=0, position=(1, 0),
                   x_sequence=time, label="speed ω")

    plotter.assign("u", column=0, row=0, position=(2, 0),
                   x_sequence=time, label="voltage V")
    plotter.assign("u_max", column=0, row=0, position=(2, 0),
                   x_sequence=time, label="voltage max", line_style="--")
    plotter.assign("u_min", column=0, row=0, position=(2, 0),
                   x_sequence=time, label="voltage min", line_style="--")

    plotter.assign("u_offset", column=0, row=0, position=(0, 1),
                   x_sequence=time, label="voltage offset")
    plotter.assign("system_noise", column=0, row=0, position=(1, 1),
                   x_sequence=time, label="system noise")

    plotter.plot()
