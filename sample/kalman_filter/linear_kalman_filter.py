"""
File: linear_kalman_filter.py

This script demonstrates the simulation and deployment of a Linear Kalman Filter (LKF) for a discrete-time linear system with input, output, and delay. It generates an input sequence using an M-sequence (LFSR), simulates the system dynamics with noise, applies the Kalman filter for state estimation, and visualizes the results. The script also shows how to generate C++ code for the Kalman filter using the KalmanFilterDeploy utility.
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np

from python_control.kalman_filter import LinearKalmanFilter

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter


def generate_m_sequence(length, taps):
    """Generate an M-sequence using a linear feedback shift register (LFSR)."""
    state = [1] * (max(taps) + 1)  # Initial state with all ones
    m_sequence = []

    for _ in range(length):
        m_sequence.append(state[-1])
        feedback = sum([state[tap] for tap in taps]) % 2
        state = [feedback] + state[:-1]

    return np.array(m_sequence)


def main():
    # Create model
    A = np.array([[1.0, 0.1, 0.0, 0.0],
                  [0.0, 1.0, 0.1, 0.0],
                  [0.0, 0.0, 1.0, 0.1],
                  [0.0, 0.0, 0.0, 1.0]])

    B = np.array([[0.0, 0.0],
                  [0.1, 0.0],
                  [0.0, 0.1],
                  [0.0, 0.0]])

    C = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]])

    Number_of_Delay = 2

    # System noise and observation noise parameters
    Q = np.diag([1.0, 1.0, 1.0, 2.0])
    R = np.eye(2) * 10.0

    # Define Kalman filter
    lkf = LinearKalmanFilter(A, B, C, Q, R, Number_of_Delay)

    # Initial state
    lkf.x_hat = np.array([[0],
                         [0],
                         [0],
                         [0]])

    # Simulation steps
    num_steps = 50

    # Generate input signal
    taps = [2, 3]
    m_sequence = generate_m_sequence(num_steps * 2, taps)
    u_data_T = m_sequence.reshape(num_steps, 2) - 0.5
    u_data = u_data_T.T

    # System noise and observation noise real
    Q_real = np.diag([1.0, 1.0, 1.0, 1.0]) * 0.0
    R_real = np.eye(2) * 0.0

    # Generate data
    np.random.seed(0)

    plotter = SimulationPlotter()

    x_true = np.array([[0.0], [0.0], [0.0], [0.1]])
    x_estimate = lkf.get_x_hat()
    y_measured = np.zeros((C.shape[0], 1))
    u = np.zeros((B.shape[1], 1))

    y_store = [np.zeros((C.shape[0], 1))] * (Number_of_Delay + 1)
    delay_index = 0

    plotter.append_name(x_true, "x_true")
    plotter.append_name(x_estimate, "x_estimate")
    plotter.append_name(y_measured, "y_measured")
    u = np.array(u_data_T.reshape(50, 2))
    plotter.append_sequence_name(u, "u")

    for k in range(1, num_steps):
        u = u_data[:, k - 1].reshape(-1, 1)

        w = np.random.multivariate_normal(np.zeros(A.shape[0]), Q_real)
        v = np.random.multivariate_normal(np.zeros(C.shape[0]), R_real)

        # system response
        x_true = A @ x_true + B @ u + w.reshape(-1, 1)
        y_store[delay_index] = C @ x_true + v.reshape(-1, 1)

        # system delay
        delay_index += 1
        if delay_index > Number_of_Delay:
            delay_index = 0

        y_measured = y_store[delay_index]

        # Kalman filter
        lkf.predict_and_update(u, y_measured)
        # x_estimate = lkf.get_x_hat()
        x_estimate = lkf.get_x_hat_without_delay()

        plotter.append_name(x_true, "x_true")
        plotter.append_name(x_estimate, "x_estimate")
        plotter.append_name(y_measured, "y_measured")

    # Kalman Gain
    print("Kalman Gain:\n", lkf.G)

    # Plot
    plotter.assign("x_true", column=0, row=0, position=(0, 0))
    plotter.assign("x_estimate", column=0, row=0, position=(0, 0))
    plotter.assign("y_measured", column=0, row=0, position=(0, 0))

    plotter.assign("x_true", column=2, row=0, position=(1, 0))
    plotter.assign("x_estimate", column=2, row=0, position=(1, 0))
    plotter.assign("y_measured", column=1, row=0, position=(1, 0))

    plotter.assign("u", column=0, row=0, position=(0, 1))

    plotter.assign("u", column=1, row=0, position=(1, 1))

    plotter.assign("x_true", column=1, row=0, position=(2, 0))
    plotter.assign("x_estimate", column=1, row=0, position=(2, 0))

    plotter.assign("x_true", column=3, row=0, position=(2, 1))
    plotter.assign("x_estimate", column=3, row=0, position=(2, 1))

    plotter.plot("True state and observation")


if __name__ == "__main__":
    main()
