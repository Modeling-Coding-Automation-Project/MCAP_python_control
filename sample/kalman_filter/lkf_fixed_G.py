"""
File: lkf_fixed_G.py

This script demonstrates the use of a Linear Kalman Filter (LKF) for state estimation in a discrete-time linear system.
It defines a simple 2-state system, configures the Kalman filter with system and noise parameters, and runs a simulation
to estimate the system states from noisy measurements. The script also generates C++ header files for the configured
Kalman filter and visualizes the true states, estimated states, control inputs, and measurements over time.
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np

from python_control.kalman_filter import LinearKalmanFilter

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter


def main():
    # Create model
    A = np.array([[0.7, 0.2],
                  [-0.3, 0.8]])
    B = np.array([[0.1],
                  [0.2]])
    C = np.array([[1.0, 0.0]])
    # D = np.array([[0.0]])

    dt = 0.01

    # System noise and observation noise parameters
    Q = np.diag([1.0, 1.0])
    R = np.diag([1.0])

    # Define Kalman filter
    lkf = LinearKalmanFilter(A, B, C, Q, R)

    # You can get converged G if you use LKF.
    lkf.converge_G()
    print("Kalman Gain:\n", lkf.G)

    # Initial state
    lkf.x_hat = np.array([[0],
                         [0]])

    # Simulation steps
    num_steps = 100
    time = np.arange(0, num_steps * dt, dt)

    # System noise and observation noise real
    Q_real = np.diag([1.0, 1.0]) * 0.0
    R_real = np.diag([1.0]) * 0.0

    # Generate data
    np.random.seed(0)

    plotter = SimulationPlotter()

    x_true = np.array([[0.1], [0.1]])
    x_estimate = lkf.get_x_hat()
    y_measured = np.zeros((C.shape[0], 1))
    u = np.ones((B.shape[1], 1))

    plotter.append_name(u, "u")
    plotter.append_name(x_true, "x_true")
    plotter.append_name(x_estimate, "x_estimate")
    plotter.append_name(y_measured, "y_measured")

    for k in range(1, num_steps):

        w = np.random.multivariate_normal(np.zeros(A.shape[0]), Q_real)
        v = np.random.multivariate_normal(np.zeros(C.shape[0]), R_real)

        # system response
        x_true = A @ x_true + B @ u + w.reshape(-1, 1)
        y_measured = C @ x_true + v.reshape(-1, 1)

        # Kalman filter
        # Using fixed G is more efficient
        lkf.predict_and_update_with_fixed_G(u, y_measured)
        x_estimate = lkf.get_x_hat()

        plotter.append_name(u, "u")
        plotter.append_name(x_true, "x_true")
        plotter.append_name(x_estimate, "x_estimate")
        plotter.append_name(y_measured, "y_measured")

    # Plot
    plotter.assign("x_true", column=0, row=0, position=(0, 0), x_sequence=time)
    plotter.assign("x_estimate", column=0, row=0,
                   position=(0, 0), x_sequence=time)
    plotter.assign("x_true", column=1, row=0, position=(1, 0), x_sequence=time)
    plotter.assign("x_estimate", column=1, row=0,
                   position=(1, 0), x_sequence=time)

    plotter.assign("y_measured", column=0, row=0,
                   position=(0, 1), x_sequence=time)

    plotter.assign("u", column=0, row=0, position=(1, 1), x_sequence=time)

    plotter.plot("True state and observation")


if __name__ == "__main__":
    main()
