"""
File: discrete_transfer_function_3_4.py

This script demonstrates the process of defining a continuous transfer function, analyzing its step response, discretizing it, and generating corresponding C++ code for the discrete transfer function. The script uses the `control` library for system modeling and response analysis, and `matplotlib` for plotting the results. It also utilizes the `TransferFunctionDeploy` class to export the discrete transfer function as C++ header files for deployment in C++ projects.
"""
import os

import sys
sys.path.append(os.getcwd())

import control
import numpy as np
import matplotlib.pyplot as plt


# Define continuous transfer function
sys = control.TransferFunction([1.0, 1.0], [1.0, 2.0, 3.0, 2.0, 1.0])
print("\nContinuous transfer function:\n", sys)

# step response
T, yout = control.step_response(sys)

# plot results
plt.plot(T, yout)
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.title('Step Response')
plt.grid(True)

# convert to discrete transfer function
sys_d = sys.sample(Ts=0.2, method='zoh')
print("\nDiscrete transfer function:\n", sys_d)

# step response
T_d, yout_d = control.step_response(sys_d)

# plot results
plt.plot(T_d, yout_d)
plt.legend(['continuous', 'discrete'])

# convert to discrete state space
sys_ss_d = control.tf2ss(sys_d)
T_ss_d, yout_ss_d = control.step_response(sys_ss_d)

# plot results
plt.plot(T_ss_d, yout_ss_d)
plt.legend(['continuous', 'discrete', 'discrete state space'])

# show y results
print("yout_d")
for i in range(len(yout_d)):
    print(yout_d[i], ",")

# show results
plt.show()
