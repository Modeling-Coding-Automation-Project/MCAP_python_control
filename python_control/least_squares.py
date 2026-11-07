"""
file: least_squares.py

Batch Least Squares (LS) and Recursive Least Squares (RLS) regression.
"""
import numpy as np


class LeastSquares:
    """
    A class for performing Least Squares regression.
    This class allows fitting a linear model to data and making predictions.
    Attributes:
        state_size (int): The number of features in the input data.
        number_of_data (int): The number of data points.
        fit_intercept (bool): Whether to append a constant column (default True).
        parameter_count (int): Number of features plus the optional intercept.
        weights (np.ndarray): Feature weights followed by the optional intercept.

    Pass raw features without a manually appended constant column. Set
    fit_intercept=False to solve min ||X weights - y||^2 through the origin.
    """

    def __init__(self, X, fit_intercept=True):
        if X.ndim != 2:
            raise ValueError("Input data must be a two-dimensional array.")
        self.state_size = X.shape[1]
        if self.state_size <= 0:
            raise ValueError("State size must be greater than 0.")

        self.number_of_data = X.shape[0]
        self._fit_intercept = bool(fit_intercept)
        self.parameter_count = self.state_size + int(self.fit_intercept)

        self.weights = np.zeros((self.parameter_count, 1), dtype=X.dtype)

    @property
    def fit_intercept(self):
        """Read-only mode; construct a new model to change parameter dimensions."""
        return self._fit_intercept

    def _design_matrix(self, X):
        """Use the same feature and intercept convention for fit and predict."""
        if X.ndim != 2 or X.shape[1] != self.state_size:
            raise ValueError("Wrong state size.")
        if self.fit_intercept:
            # Preserve the existing float64 promotion when adding an intercept.
            return np.hstack([X, np.ones((X.shape[0], 1))])
        return X

    def fit(self, X, y):
        """
        Fit the model to the provided data using Least Squares method.
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): Targets of shape (n_samples,) or (n_samples, 1).
        Raises:
            ValueError: If the input data does not match the expected state size.
        """
        self.weights = np.linalg.lstsq(
            self._design_matrix(X), y, rcond=None)[0]

    def predict(self, X):
        """
        Predict the target values for the given input data.
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
        Returns:
            np.ndarray: Predictions with the same target dimensionality as fit.
        Raises:
            ValueError: If the input data does not match the expected state size.
        """
        return self._design_matrix(X) @ self.weights

    def get_weights(self):
        """
        Get the weights of the fitted model.
        Returns:
            np.ndarray: Feature weights, followed by a bias only when
                fit_intercept=True. A one-dimensional target gives a vector;
                a column target gives a column of weights.
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
