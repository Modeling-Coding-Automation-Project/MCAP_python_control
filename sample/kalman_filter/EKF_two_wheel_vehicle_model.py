"""
Extended Kalman Filter for Two-Wheel Vehicle Model

This module implements an Extended Kalman Filter (EKF) for estimating the state
of a two-wheel vehicle model. It uses symbolic mathematics to derive the state
and measurement functions, along with their Jacobians, and then deploys them
as executable Python functions.

The vehicle model includes:
- Lateral dynamics (beta, beta_dot)
- Yaw dynamics (r, r_dot)
- Longitudinal dynamics (V, accel)
- Position tracking (px, py)
- Orientation tracking (theta)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

import math
import numpy as np
import sympy as sp
from dataclasses import dataclass

from python_control.kalman_filter import ExtendedKalmanFilter
from python_control.control_deploy import ExpressionDeploy

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter
from sample.simulation_manager.signal_edit.sampler import PulseGenerator


def create_model(delta_time: float):
    # define parameters and variables
    m, u, v, r, F_f, F_r = sp.symbols('m u v r F_f F_r', real=True)
    I, l_f, l_r, v_dot, r_dot, V, beta, beta_dot = sp.symbols(
        'I l_f l_r v_dot r_dot V beta beta_dot', real=True)

    # derive equations of two wheel vehicle model
    eq_1 = sp.Eq(m * (v_dot + u * r), F_f + F_r)
    eq_2 = sp.Eq(I * r_dot, l_f * F_f - l_r * F_r)

    lhs = eq_1.lhs.subs({u: V, v_dot: V * beta_dot})
    eq1 = sp.Eq(lhs, eq_1.rhs)

    K_f, K_r, delta, beta_f, beta_r = sp.symbols(
        'K_f K_r delta beta_f beta_r', real=True)

    rhs = eq1.rhs.subs({F_f: -2 * K_f * beta_f, F_r: -2 * K_r * beta_r})
    eq_1 = sp.Eq(eq1.lhs, rhs)

    rhs = eq_2.rhs.subs({F_f: -2 * K_f * beta_f, F_r: -2 * K_r * beta_r})
    eq_2 = sp.Eq(eq_2.lhs, rhs)

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

    solution = sp.solve(eq_vec, r_dot, dict=True)
    r_dot_sol = sp.simplify(solution[0][r_dot])

    # Define state space model
    accel = sp.symbols('accel', real=True)
    U = sp.Matrix([[delta], [accel]])

    theta, px, py = sp.symbols('theta px py', real=True)
    X = sp.Matrix([[px], [py], [theta], [r], [beta], [V]])
    Y = sp.Matrix([[px], [py], [theta], [r], [V]])

    fxu_continuous = sp.Matrix([
        [V * sp.cos(theta)],
        [V * sp.sin(theta)],
        [r],
        [r_dot_sol],
        [beta_dot_sol],
        [accel],
    ])
    fxu: sp.Matrix = X + fxu_continuous * delta_time

    print("State Function (fxu):")
    sp.pprint(fxu)

    hx = sp.Matrix([[X[0]], [X[1]], [X[2]], [X[3]], [X[5]]])
    print("Measurement Function (hx):")
    sp.pprint(hx)

    # derive Jacobian
    fxu_jacobian = fxu.jacobian(X)
    hx_jacobian = hx.jacobian(X)

    fxu_file_name = ExpressionDeploy.write_state_function_code_from_sympy(
        fxu, X, U)
    fxu_jacobian_file_name = \
        ExpressionDeploy.write_state_function_code_from_sympy(
            fxu_jacobian, X, U)

    hx_file_name = ExpressionDeploy.write_measurement_function_code_from_sympy(
        hx, X)
    hx_jacobian_file_name = \
        ExpressionDeploy.write_measurement_function_code_from_sympy(
            hx_jacobian, X)

    return X, U, Y, \
        fxu_file_name, fxu_jacobian_file_name, \
        hx_file_name, hx_jacobian_file_name


@dataclass
class Parameter:
    m: float = 2000
    l_f: float = 1.4
    l_r: float = 1.6
    I: float = 4000
    K_f: float = 12e3
    K_r: float = 11e3


def main():
    # simulation setup
    sim_delta_time = 0.01
    simulation_time = 20.0

    time = np.arange(0, simulation_time, sim_delta_time)

    X, U, Y, \
        fxu_file_name, fxu_jacobian_file_name, \
        hx_file_name, hx_jacobian_file_name = create_model(sim_delta_time)

    parameters_ekf = Parameter()

    Q_ekf = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    R_ekf = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])

    local_vars = {}

    exec(f"from {fxu_file_name} import function as fxu_script_function",
         globals(), local_vars)
    exec(
        f"from {fxu_jacobian_file_name} import function as fxu_jacobian_script_function", globals(), local_vars)
    exec(f"from {hx_file_name} import function as hx_script_function",
         globals(), local_vars)
    exec(
        f"from {hx_jacobian_file_name} import function as hx_jacobian_script_function", globals(), local_vars)

    fxu_script_function = local_vars["fxu_script_function"]
    fxu_jacobian_script_function = local_vars["fxu_jacobian_script_function"]
    hx_script_function = local_vars["hx_script_function"]
    hx_jacobian_script_function = local_vars["hx_jacobian_script_function"]

    ekf = ExtendedKalmanFilter(
        state_function=fxu_script_function,
        measurement_function=hx_script_function,
        state_function_jacobian=fxu_jacobian_script_function,
        measurement_function_jacobian=hx_jacobian_script_function,
        Q=Q_ekf,
        R=R_ekf,
        Parameters=parameters_ekf
    )

    # X: px, py, theta, r, beta, V
    x_true = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])

    ekf.x_hat = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.5]])

    # U: delta, accel
    u = np.array([[0.0], [0.0]])

    # create delta sequence
    _, input_signal = PulseGenerator.sample_pulse(
        sampling_interval=sim_delta_time,
        start_time=time[0],
        period=6.0,
        pulse_width=50.0,
        pulse_amplitude=2.0,
        duration=time[-1],
    )
    input_signal = input_signal - 1.0
    delta_sequence = np.zeros_like(time)

    delta_max = 30.0 / 180.0 / math.pi
    for i in range(len(time)):
        if i == 0:
            delta_sequence[i] = input_signal[i, 0] * delta_max * sim_delta_time
        else:
            delta_sequence[i] = delta_sequence[i - 1] + \
                input_signal[i, 0] * delta_max * sim_delta_time

    # create sequence for acceleration
    _, signal_plus = PulseGenerator.sample_pulse(
        sampling_interval=sim_delta_time,
        start_time=time[0],
        period=10.0,
        pulse_width=25.0,
        pulse_amplitude=1.0,
        duration=time[-1],
    )

    _, signal_minus = PulseGenerator.sample_pulse(
        sampling_interval=sim_delta_time,
        start_time=time[0] + 5.0,
        period=10.0,
        pulse_width=25.0,
        pulse_amplitude=-1.0,
        duration=time[-1],
    )

    accel_sequence = np.zeros_like(time)
    for i in range(len(time)):
        accel_sequence[i] = signal_plus[i, 0] + signal_minus[i, 0]

    plotter = SimulationPlotter()

    # simulation
    for i in range(round(simulation_time / sim_delta_time)):
        u = np.array([[delta_sequence[i]], [accel_sequence[i]]])

        # system response
        x_true = fxu_script_function(x_true, u, parameters_ekf)
        y_measured = hx_script_function(x_true, parameters_ekf)

        # estimate
        ekf.predict(u)
        ekf.update(y_measured)

        x_estimated = ekf.get_x_hat_without_delay()

        plotter.append_name(x_true, "x_true")
        plotter.append_name(x_estimated, "x_estimated")
        plotter.append_name(y_measured, "y_measured")
        plotter.append_name(u, "u")

    # plot
    plotter.assign("x_true", column=0, row=0, position=(0, 0),
                   x_sequence=time, label="px_true")
    plotter.assign("x_estimated", column=0, row=0, position=(0, 0),
                   x_sequence=time, label="px_estimated")

    plotter.assign("x_true", column=1, row=0, position=(1, 0),
                   x_sequence=time, label="py_true")
    plotter.assign("x_estimated", column=1, row=0, position=(1, 0),
                   x_sequence=time, label="py_estimated")

    plotter.assign("x_true", column=2, row=0, position=(2, 0),
                   x_sequence=time, label="theta_true")
    plotter.assign("x_estimated", column=2, row=0, position=(2, 0),
                   x_sequence=time, label="theta_estimated")

    plotter.assign("x_true", column=3, row=0, position=(0, 1),
                   x_sequence=time, label="r_true")
    plotter.assign("x_estimated", column=3, row=0, position=(0, 1),
                   x_sequence=time, label="r_estimated")

    plotter.assign("x_true", column=4, row=0, position=(1, 1),
                   x_sequence=time, label="beta_true")
    plotter.assign("x_estimated", column=4, row=0, position=(1, 1),
                   x_sequence=time, label="beta_estimated")

    plotter.assign("x_true", column=5, row=0, position=(2, 1),
                   x_sequence=time, label="V_true")
    plotter.assign("x_estimated", column=5, row=0, position=(2, 1),
                   x_sequence=time, label="V_estimated")

    plotter.assign("u", column=0, row=0, position=(0, 2),
                   x_sequence=time, label="delta")
    plotter.assign("u", column=1, row=0, position=(1, 2),
                   x_sequence=time, label="a")

    plotter.plot()


if __name__ == "__main__":
    main()
