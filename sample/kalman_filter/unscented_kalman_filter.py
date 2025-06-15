"""
File: unscented_kalman_filter.py

This script demonstrates the implementation and simulation of an Unscented Kalman Filter (UKF) for a bicycle model with delayed measurements. The code defines the nonlinear state transition and measurement functions using symbolic computation, generates deployable code for these functions, and sets up a UKF to estimate the state of a simulated vehicle moving in a 2D plane with two landmarks as references. The simulation includes process and measurement noise, as well as configurable system delay. Results are visualized using a plotting utility.

Reference URL:
https://inzkyk.xyz/kalman_filter/unscented_kalman_filter/#section:10.15
"""
import os
import sys
sys.path.append(os.getcwd())

import math
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import symbols

from python_control.kalman_filter import UnscentedKalmanFilter_Basic, UnscentedKalmanFilter
from python_control.control_deploy import ExpressionDeploy

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter


def main():
    # %% bicycle model example
    # state X: [x, y, theta]
    # input U: [v, steering_angle]
    # output Y: [r_p, angle_p]

    delta_time = symbols('delta_time')
    steering_angle = symbols('steering_angle')
    x = symbols('x')
    y = symbols('y')
    v = symbols('v')
    wheelbase = symbols('wheelbase')
    radius = symbols('radius')
    theta = symbols('theta')
    landmark_1_x = symbols('landmark_1_x')
    landmark_1_y = symbols('landmark_1_y')
    landmark_2_x = symbols('landmark_2_x')
    landmark_2_y = symbols('landmark_2_y')

    # define state X, input U
    X = sympy.Matrix([[x], [y], [theta]])
    U = sympy.Matrix([[v], [steering_angle]])

    d = v * delta_time
    beta = (d / wheelbase) * sympy.tan(steering_angle)
    r = wheelbase / sympy.tan(steering_angle)

    fxu = sympy.Matrix([[x - r * sympy.sin(theta) + r * sympy.sin(theta + beta)],
                        [y + r * sympy.cos(theta) - r *
                         sympy.cos(theta + beta)],
                        [theta + beta]])

    hx = sympy.Matrix([[sympy.sqrt((landmark_1_x - x) ** 2 + (landmark_1_y - y) ** 2)],
                       [sympy.atan2(landmark_1_y - y,
                                    landmark_1_x - x) - theta],
                       [sympy.sqrt((landmark_2_x - x) ** 2 +
                                   (landmark_2_y - y) ** 2)],
                       [sympy.atan2(landmark_2_y - y, landmark_2_x - x) - theta]])

    # Save functions to separate files
    ExpressionDeploy.write_state_function_code_from_sympy(fxu, X, U)
    ExpressionDeploy.write_measurement_function_code_from_sympy(hx, X)

    # %% design EKF

    landmarks = np.array([[-1.0, 10.0], [-1.0, 10.0]])

    class Parameters:
        def __init__(self, delta_time, wheelbase,
                     landmark_1_x, landmark_1_y, landmark_2_x, landmark_2_y):
            self.delta_time = delta_time
            self.wheelbase = wheelbase
            self.landmark_1_x = landmark_1_x
            self.landmark_1_y = landmark_1_y
            self.landmark_2_x = landmark_2_x
            self.landmark_2_y = landmark_2_y

    Parameters_ukf = Parameters(
        delta_time=0.1, wheelbase=0.5,
        landmark_1_x=landmarks[0, 0], landmark_1_y=landmarks[1, 0],
        landmark_2_x=landmarks[0, 1], landmark_2_y=landmarks[1, 1])

    Number_of_Delay = 5

    Q_ukf = np.diag([0.01, 0.01, 0.01])
    R_ukf = np.diag([1.0, 1.0, 1.0, 1.0])

    import fxu
    import hx
    # ukf = UnscentedKalmanFilter_Basic(fxu.function, hx.function,
    #                                   Q_ukf, R_ukf, Parameters_ukf,
    #                                   Number_of_Delay, kappa=0.5)

    ukf = UnscentedKalmanFilter(fxu.function, hx.function,
                                Q_ukf, R_ukf, Parameters_ukf,
                                Number_of_Delay)

    # %% bicycle model simulation

    sim_delta_time = 0.1
    sim_wheelbase = Parameters_ukf.wheelbase
    simulation_time = 20.0

    def move(x, u, dt, wheelbase):
        hdg = x[2, 0]
        vel = u[0, 0]
        steering_angle = u[1, 0]
        dist = vel * dt

        if abs(steering_angle) > 0.001:  # is robot rotating?
            beta = (dist / wheelbase) * math.tan(steering_angle)
            r = wheelbase / math.tan(steering_angle)  # radius

            dx = np.array([[-r * math.sin(hdg) + r * math.sin(hdg + beta)],
                           [r * math.cos(hdg) - r * math.cos(hdg + beta)],
                           [beta]])
        else:  # moving in straight line
            dx = np.array([[dist * math.cos(hdg)],
                           [dist * math.sin(hdg)],
                           [0]])

        x_next = x + dx + np.array([[np.random.randn() * 0.0],
                                    [np.random.randn() * 0.0],
                                    [np.random.randn() * 0.0]])

        theta = x_next[2, 0]
        # landmark 1
        x_dif = (landmarks[0, 0] - x_next[0, 0])
        y_dif = (landmarks[1, 0] - x_next[1, 0])
        r_1 = math.sqrt(x_dif * x_dif + y_dif * y_dif) + \
            np.random.randn() * 0.0
        phi_1 = math.atan2(y_dif, x_dif) - theta + np.random.randn() * 0.0

        # landmark 2
        x_dif = (landmarks[0, 1] - x_next[0, 0])
        y_dif = (landmarks[1, 1] - x_next[1, 0])
        r_2 = math.sqrt(x_dif * x_dif + y_dif * y_dif) + \
            np.random.randn() * 0.0
        phi_2 = math.atan2(y_dif, x_dif) - theta + np.random.randn() * 0.0

        y_next = np.array([[r_1], [phi_1], [r_2], [phi_2]])

        return x_next, y_next

    def run_simulation():

        x_true = np.array([[2.0], [6.0], [0.3]])  # initial x, y, theta
        u = np.array([[2.0], [0.1]])  # inputs v, steering_angle

        ukf.x_hat = np.array([[0.0], [0.0], [0.0]])  # initial state

        plotter = SimulationPlotter()

        time = np.arange(0, simulation_time, sim_delta_time)

        y_store = [np.zeros((R_ukf.shape[0], 1))] * (Number_of_Delay + 1)
        delay_index = 0

        for i in range(round(simulation_time / sim_delta_time)):
            x_true, y_store[delay_index] = move(x_true, u, sim_delta_time,
                                                sim_wheelbase)  # simulate robot

            # system delay
            delay_index += 1
            if delay_index > Number_of_Delay:
                delay_index = 0

            y_measured = y_store[delay_index]

            # estimate
            ukf.predict(u)
            ukf.update(y_store[delay_index])

            x_estimated = ukf.get_x_hat_without_delay()

            plotter.append(x_true)
            plotter.append(x_estimated)
            plotter.append(y_measured)
            plotter.append(u)

        # plot
        plotter.assign("x_true", column=0, row=0, position=(0, 0),
                       x_sequence=time, label="px_true")
        plotter.assign("x_estimated", column=0, row=0, position=(0, 0),
                       x_sequence=time, label="px_estimated")

        plotter.assign("x_true", column=1, row=0, position=(1, 0),
                       x_sequence=time, label="py_true")
        plotter.assign("x_estimated", column=1, row=0, position=(1, 0),
                       x_sequence=time, label="px_estimated")

        plotter.assign("x_true", column=2, row=0, position=(2, 0),
                       x_sequence=time, label="theta_true")
        plotter.assign("x_estimated", column=2, row=0, position=(2, 0),
                       x_sequence=time, label="theta_estimated")

        plotter.assign("u", column=0, row=0, position=(0, 1),
                       x_sequence=time, label="v")
        plotter.assign("u", column=1, row=0, position=(0, 1),
                       x_sequence=time, label="steering_angle")

        plotter.assign("y_measured", column=0, row=0, position=(1, 1),
                       x_sequence=time, label="r_landmark_1")
        plotter.assign("y_measured", column=2, row=0, position=(1, 1),
                       x_sequence=time, label="r_landmark_2")

        plotter.assign("y_measured", column=1, row=0, position=(2, 1),
                       x_sequence=time, label="angle_landmark_1")
        plotter.assign("y_measured", column=3, row=0, position=(2, 1),
                       x_sequence=time, label="angle_landmark_2")

        plotter.plot("UKF for bicycle model results")

    run_simulation()

    plt.show()


if __name__ == "__main__":
    main()
