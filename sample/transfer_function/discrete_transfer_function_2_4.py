"""
file: discrete_transfer_function_2_4.py

This script demonstrates the creation, analysis, and deployment of a discrete transfer function using the `python-control` library.
It defines a discrete transfer function, generates corresponding C++ header code for deployment, and compares the step response
of the transfer function in both transfer function and state-space representations. The results are visualized using matplotlib.
"""
import os

import sys
sys.path.append(os.getcwd())

import control
import numpy as np
import matplotlib.pyplot as plt

# Define discrete transfer function
sys_d = control.TransferFunction([0.5, 0.3, 0.1], [
    1.0, -1.8, 1.5, -0.7, 0.2], dt=0.2)
print("\nDiscrete transfer function:\n", sys_d)

# step response
T, yout = control.step_response(sys_d)

# plot results
plt.plot(T, yout)
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.title('Step Response')
plt.grid(True)

# convert to discrete state space
sys_ss_d = control.tf2ss(sys_d)

# step response
T_ss_d, yout_ss_d = control.step_response(sys_ss_d)

# plot results
plt.plot(T_ss_d, yout_ss_d)
plt.legend(['discrete', 'discrete state space'])

# show y results
print("yout_ss_d")
for i in range(len(yout_ss_d)):
    print(yout_ss_d[i], ",")

# show results
plt.show()
