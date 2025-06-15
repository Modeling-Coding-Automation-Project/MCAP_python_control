"""
file: least_squares.py

This module implements a Recursive Least Squares (RLS) algorithm for online linear regression.
The RLS algorithm incrementally updates model parameters to minimize the squared error between
predicted and true values, making it suitable for adaptive filtering and real-time system identification.
"""
import numpy as np


class LeastSquares:
    """
    A class for performing Least Squares regression.
    This class allows fitting a linear model to data and making predictions.
    Attributes:
        state_size (int): The number of features in the input data.
        number_of_data (int): The number of data points.
        weights (np.ndarray): The weights of the linear model, including a bias term.
    """

    def __init__(self, X):
        self.state_size = X.shape[1]
        if self.state_size <= 0:
            raise ValueError("State size must be greater than 0.")

        self.number_of_data = X.shape[0]

        self.weights = np.zeros((self.state_size + 1, 1), dtype=X.dtype)

    def fit(self, X, y):
        """
        Fit the model to the provided data using Least Squares method.
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        Raises:
            ValueError: If the input data does not match the expected state size.
        """
        if X.shape[1] != self.state_size:
            raise ValueError("Wrong state size.")

        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term

        self.weights = np.linalg.lstsq(X, y)[0]

    def predict(self, X):
        """
        Predict the target values for the given input data.
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
        Returns:
            np.ndarray: Predicted target values of shape (n_samples,).
        Raises:
            ValueError: If the input data does not match the expected state size.
        """
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term

        return X @ self.weights

    def get_weights(self):
        """
        Get the weights of the fitted model.
        Returns:
            np.ndarray: The weights of the model, including the bias term.
        """
        return self.weights


class RecursiveLeastSquares:
    """
    A class for performing Recursive Least Squares (RLS) regression.
    This class allows for online linear regression, updating model parameters
    incrementally as new data arrives.
    Attributes:
        RLS_size (int): The number of features in the input data plus one for the bias term.
        lambda_factor (float): The forgetting factor for the RLS algorithm.
        delta (float): A small value to initialize the inverse covariance matrix.
        P (np.ndarray): The inverse covariance matrix used in the RLS algorithm.
        weights (np.ndarray): The weights of the linear model, including a bias term.
    """

    def __init__(self, feature_size, lambda_factor=0.9, delta=0.1):
        self.RLS_size = feature_size + 1  # Add bias term
        self.lambda_factor = lambda_factor
        self.delta = delta
        self.P = np.eye(self.RLS_size) * delta
        self.weights = np.zeros((self.RLS_size, 1))

    def update(self, x, y_true):
        """
        Update the model with a new data point using the RLS algorithm.
        Args:
            x (np.ndarray): Input data of shape (n_features,).
            y_true (float): The true target value for the input data.
        Raises:
            ValueError: If the input data does not match the expected feature size.
        """
        Ones = np.ones((1, 1))  # Add bias term
        x_ex = np.vstack((x, Ones))

        y = x_ex.T @ self.weights
        y_dif = y_true - y

        P_x = self.P @ x_ex
        K = P_x / (self.lambda_factor + x_ex.T @ P_x)

        self.weights += K @ y_dif
        self.P = (self.P - K @ x_ex.T @ self.P) / self.lambda_factor

    def predict(self, X):
        """
        Predict the target values for the given input data using the RLS model.
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
        Returns:
            np.ndarray: Predicted target values of shape (n_samples,).
        Raises:
            ValueError: If the input data does not match the expected feature size.
        """
        ones = np.ones((X.shape[0], 1))
        return np.hstack((X, ones)) @ self.weights

    def get_weights(self):
        """
        Get the weights of the fitted RLS model.
        Returns:
            np.ndarray: The weights of the model, including the bias term.
        """
        return self.weights
