"""
File: recursive_squares_method.py

This script demonstrates the use of Recursive Least Squares (RLS) for online parameter estimation in a linear regression setting. It generates synthetic data, applies the RLS algorithm to estimate the weights of a linear model, and visualizes the results. The script also shows how to export the RLS model as C++ code for deployment.
"""
import os
import sys
sys.path.append(os.getcwd())

import math
import numpy as np
import copy
import matplotlib.pyplot as plt

from python_control.least_squares import RecursiveLeastSquares

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter

# Create data
np.random.seed(42)
n_samples = 100
x1 = np.random.rand(n_samples) * 10
x2 = np.random.rand(n_samples) * 10
offset = np.random.normal(0.3, 0.01, n_samples)

weights_true = np.array([[0.5], [-0.2], [0.3]])

y = weights_true[0, 0] * x1 + weights_true[1, 0] * x2 + offset
X = np.column_stack((x1, x2))

# Create Recursive Least Squares object
rls = RecursiveLeastSquares(feature_size=X.shape[1], lambda_factor=0.9)

plotter = SimulationPlotter()

for i in range(n_samples):
    x = X[i].reshape(-1, 1)
    y_true = y[i].reshape(-1, 1)

    rls.update(x, y_true)

    weights_predicted = copy.deepcopy(rls.get_weights())
    plotter.append(weights_predicted)
    plotter.append(weights_true)

predictions = rls.predict(X)

print("true weights:", weights_true.T)
print("predicted weights:", rls.get_weights().T)
print("true y:", y[:5])
print("predicted y:", predictions[:5].T)

# Plot the results
fig, axs = plt.subplots(3, 1)
axs[0].plot(x1, label="x1")
axs[0].legend()
axs[0].grid(True)
axs[1].plot(x2, label="x2")
axs[1].legend()
axs[1].grid(True)
axs[2].plot(offset, label="offset")
axs[2].legend()
axs[2].grid(True)


plotter.assign("weights_predicted", column=0, row=0, position=(0, 0))
plotter.assign("weights_true", column=0, row=0, position=(0, 0))
plotter.assign("weights_predicted", column=1, row=0, position=(1, 0))
plotter.assign("weights_true", column=1, row=0, position=(1, 0))
plotter.assign("weights_predicted", column=2, row=0, position=(2, 0))
plotter.assign("weights_true", column=2, row=0, position=(2, 0))

plotter.plot("Discrete-Time State-Space Response")
