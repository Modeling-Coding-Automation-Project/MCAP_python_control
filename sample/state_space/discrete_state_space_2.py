"""
File: discrete_state_space_2.py

This script demonstrates the creation, analysis, and deployment of a state-space model for a dynamic system using Python.
It defines a continuous-time state-space system, simulates its step response, discretizes the system, and generates C++ header files for deployment.
The script also visualizes the step responses for both continuous and discrete systems.
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import control
import matplotlib.pyplot as plt


def plot_y_response(T, y):
    y_0 = y[0][0]
    y_1 = y[1][0]

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(T, y_0)
    plt.xlabel('Time (s)')
    plt.ylabel('Response 0')
    plt.title('Step Response 0')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(T, y_1)
    plt.xlabel('Time (s)')
    plt.ylabel('Response 1')
    plt.title('Step Response 1')
    plt.grid(True)

    plt.tight_layout()


# define continuous state-space model
A = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [-51.2077076817131, -1.0, 2.56038538408566, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [128.019900633784, 0.0, -6.40099503168920, -10.2]
])

B = np.array([
    [0.0],
    [0.0],
    [0.0],
    [1.0]
])
C = np.array([
    [1, 0, 0, 0],
    [1280.19900633784, 0, -64.0099503168920, 0]
])
D = np.array([
    [0],
    [0]
])

sys = control.StateSpace(A, B, C, D)
T, yout = control.step_response(sys)

plot_y_response(T, yout)

# convert to discrete state-space model
dt = 0.01
sys_d = sys.sample(Ts=dt, method='euler')
T_d, yout_d = control.step_response(sys_d)

plot_y_response(T_d, yout_d)

# show A, B, C, D
print("\nA:\n", sys_d.A)
print("\nB:\n", sys_d.B)
print("\nC:\n", sys_d.C)
print("\nD:\n", sys_d.D)

# show yout_d
y_0 = yout_d[0][0]
y_1 = yout_d[1][0]
print("\nyout:\n")
for i in range(100):
    print(y_0[i], ", ", y_1[i], ",")

# show results
plt.show()
