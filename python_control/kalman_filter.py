"""
File: kalman_filter.py

This module provides implementations of several types of Kalman Filters for state estimation in control systems and signal processing. It supports linear, extended (nonlinear), and unscented (nonlinear, non-Gaussian) Kalman Filters, with optional support for input delay handling.
"""
import math
import numpy as np
import copy

KALMAN_FILTER_DIVISION_MIN = 1e-10
LKF_G_CONVERGE_REPEAT_MAX = 1000


class DelayedVectorObject:
    """
    A class to manage a fixed-length buffer (delay line) of vectors, supporting delayed access and circular indexing.

    Attributes:
        store (np.ndarray): 2D array storing vectors for each delay step.
        delay_index (int): Current index for the next vector insertion.
        Number_of_Delay (int): Number of delay steps (buffer size - 1).
    """

    def __init__(self, size, Number_of_Delay):
        self.store = np.zeros((size, Number_of_Delay + 1))
        self.delay_index = 0
        self.Number_of_Delay = Number_of_Delay

    def push(self, vector: np.ndarray):
        """
        Push a new vector into the delay line, replacing the oldest vector.
        Args:
            vector (np.ndarray): The vector to be added, must match the size of the store.
        """
        self.store[:, self.delay_index] = vector.flatten()

        self.delay_index += 1
        if self.delay_index > self.Number_of_Delay:
            self.delay_index = 0

    def get(self):
        """
        Get the current vector from the delay line, which is the most recent vector added.
        Returns:
            np.ndarray: The most recent vector in the delay line.
        """
        return self.store[:, self.delay_index].reshape(-1, 1)

    def get_by_index(self, index):
        """
        Get a vector from the delay line by index.
        Args:
            index (int): The index of the vector to retrieve, where 0 is the most recent.
        Returns:
            np.ndarray: The vector at the specified index in the delay line.
        """
        if index > self.Number_of_Delay:
            index = self.Number_of_Delay

        return self.store[:, index].reshape(-1, 1)

    def get_latest(self):
        """
        Get the latest vector from the delay line, which is the most recent vector added.
        Returns:
            np.ndarray: The latest vector in the delay line.
        """
        index = self.delay_index
        if index == 0:
            index = self.Number_of_Delay
        else:
            index = index - 1

        return self.store[:, index].reshape(-1, 1)


class KalmanFilterCommon:
    """
    KalmanFilterCommon provides a base implementation for a Kalman filter with optional input delay handling.

    Attributes:
        Number_of_Delay (int): The number of input delays to consider in the filter.
        _input_count (int): Internal counter for the number of inputs processed.
    """

    def __init__(self, Number_of_Delay=0):
        self.Number_of_Delay = Number_of_Delay
        self._input_count = 0

    def predict_and_update(self, u: np.ndarray, y: np.ndarray):
        """
        Predict the next state based on the input and update the state estimate with the measurement.
        Args:
            u (np.ndarray): Input vector for the prediction step.
            y (np.ndarray): Measurement vector for the update step.
        """
        self.predict(u)
        self.update(y)

    def get_x_hat(self):
        """
        Get the current state estimate.
        Returns:
            np.ndarray: The current state estimate vector.
        """
        return self.x_hat


class LinearKalmanFilter(KalmanFilterCommon):
    """
    LinearKalmanFilter implements a standard Kalman filter for linear systems.
    It uses matrices A, B, C for state transition, control input, and measurement respectively,
    along with process noise covariance Q and measurement noise covariance R.
    Attributes:
        A (np.ndarray): State transition matrix.
        B (np.ndarray): Control input matrix.
        C (np.ndarray): Measurement matrix.
        Q (np.ndarray): Process noise covariance matrix.
        R (np.ndarray): Measurement noise covariance matrix.
        x_hat (np.ndarray): Current state estimate.
        P (np.ndarray): Estimate error covariance.
        G (np.ndarray): Kalman gain.
        u_store (DelayedVectorObject): Object to handle delayed input vectors.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray,
                 C: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 Number_of_Delay=0):
        super().__init__(Number_of_Delay)
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.x_hat = np.zeros((A.shape[0], 1))
        self.P = np.eye(A.shape[0])
        self.G = None

        self.u_store = DelayedVectorObject(B.shape[1], Number_of_Delay)

    def predict(self, u: np.ndarray):
        """
        Predict the next state based on the current state and input.
        Args:
            u (np.ndarray): Input vector for the prediction step.
        """
        self.u_store.push(u)

        if self._input_count < self.Number_of_Delay:
            self._input_count += 1
        else:
            self.x_hat = self.A @ self.x_hat + self.B @ self.u_store.get()
            self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y: np.ndarray):
        """
        Update the state estimate with the measurement.
        Args:
            y (np.ndarray): Measurement vector for the update step.
        """
        P_CT_matrix = self.P @ self.C.T

        S_matrix = self.C @ P_CT_matrix + self.R
        self.G = P_CT_matrix @ np.linalg.inv(S_matrix)
        self.x_hat = self.x_hat + self.G @ (y - self.C @ self.x_hat)

        self.P = (np.eye(self.A.shape[0]) - self.G @ self.C) @ self.P

    def get_x_hat_without_delay(self):
        """
        Get the current state estimate without considering input delays.
        Returns:
            np.ndarray: The current state estimate vector without delay.
        """
        if self.Number_of_Delay == 0:
            return self.x_hat
        else:
            x_hat = copy.deepcopy(self.x_hat)
            delay_index = self.u_store.delay_index + \
                self.Number_of_Delay - self._input_count

            for i in range(self._input_count):
                delay_index += 1
                if delay_index > self.Number_of_Delay:
                    delay_index = delay_index - self.Number_of_Delay - 1

                x_hat = self.A @ x_hat + \
                    self.B @ self.u_store.get_by_index(delay_index)

            return x_hat

    # If G is known, you can use below "_fixed_G" functions.
    def predict_with_fixed_G(self, u: np.ndarray):
        """
        Predict the next state using a fixed Kalman gain G.
        Args:
            u (np.ndarray): Input vector for the prediction step.
        """
        self.u_store.push(u)

        if self._input_count < self.Number_of_Delay:
            self._input_count += 1
        else:
            self.x_hat = self.A @ self.x_hat + self.B @ self.u_store.get()

    def update_with_fixed_G(self, y: np.ndarray):
        """
        Update the state estimate with the measurement using a fixed Kalman gain G.
        Args:
            y (np.ndarray): Measurement vector for the update step.
        """
        self.x_hat = self.x_hat + self.G @ (y - self.C @ self.x_hat)

    def predict_and_update_with_fixed_G(self, u: np.ndarray, y: np.ndarray):
        """
        Predict the next state and update the state estimate using a fixed Kalman gain G.
        Args:
            u (np.ndarray): Input vector for the prediction step.
            y (np.ndarray): Measurement vector for the update step.
        """
        self.predict_with_fixed_G(u)
        self.update_with_fixed_G(y)

    def update_P_one_step(self):
        """
        Update the estimate error covariance P one step, assuming G is already calculated.
        This method is used to iteratively refine the covariance estimate.
        """
        self.P = self.A @ self.P @ self.A.T + self.Q

        P_CT_matrix = self.P @ self.C.T
        S_matrix = self.C @ P_CT_matrix + self.R
        self.G = P_CT_matrix @ np.linalg.inv(S_matrix)

        self.P = (np.eye(self.A.shape[0]) - self.G @ self.C) @ self.P

    def converge_G(self):
        """
        Iteratively refine the Kalman gain G until convergence.
        This method checks if the change in G is below a certain threshold for all elements.
        It is useful when G is not known initially and needs to be estimated.
        """
        for k in range(LKF_G_CONVERGE_REPEAT_MAX):
            if self.G is None:
                self.update_P_one_step()

            previous_G = self.G
            self.update_P_one_step()
            G_diff = self.G - previous_G

            is_converged = True
            for i in range(self.G.shape[0]):
                for j in range(self.G.shape[1]):

                    if (abs(self.G[i, j]) > KALMAN_FILTER_DIVISION_MIN) and \
                            (abs(G_diff[i, j] / self.G[i, j]) > KALMAN_FILTER_DIVISION_MIN):
                        is_converged = False

            if is_converged:
                break


class ExtendedKalmanFilter(KalmanFilterCommon):
    """
    ExtendedKalmanFilter implements an extended Kalman filter for nonlinear systems.
    It uses nonlinear state and measurement functions, along with their Jacobians.
    Attributes:
        state_function (callable): Nonlinear state transition function.
        measurement_function (callable): Nonlinear measurement function.
        state_function_jacobian (callable): Jacobian of the state function.
        measurement_function_jacobian (callable): Jacobian of the measurement function.
        A (np.ndarray): State transition Jacobian matrix.
        C (np.ndarray): Measurement Jacobian matrix.
        Q (np.ndarray): Process noise covariance matrix.
        R (np.ndarray): Measurement noise covariance matrix.
        x_hat (np.ndarray): Current state estimate.
        P (np.ndarray): Estimate error covariance.
        G (np.ndarray): Kalman gain.
        u_store (DelayedVectorObject): Object to handle delayed input vectors.
    """

    def __init__(self, state_function, measurement_function,
                 state_function_jacobian, measurement_function_jacobian,
                 Q: np.ndarray, R: np.ndarray, Parameters=None, Number_of_Delay=0):
        super().__init__(Number_of_Delay)
        self.state_function = state_function
        self.measurement_function = measurement_function
        self.state_function_jacobian = state_function_jacobian
        self.measurement_function_jacobian = measurement_function_jacobian

        self.A = np.zeros(Q.shape)
        self.C = np.zeros((R.shape[0], Q.shape[0]))
        self.Q = Q
        self.R = R

        self.x_hat = np.zeros((Q.shape[0], 1))
        self.P = np.eye(Q.shape[0])
        self.G = None

        self.Parameters = Parameters
        self.u_store = None

    def predict(self, u: np.ndarray):
        """
        Predict the next state based on the current state and input.
        Args:
            u (np.ndarray): Input vector for the prediction step.
        """
        if self.u_store is None:
            self.u_store = DelayedVectorObject(
                u.shape[0], self.Number_of_Delay)

        self.u_store.push(u)

        if self._input_count < self.Number_of_Delay:
            self._input_count += 1
        else:
            self.A = self.state_function_jacobian(
                self.x_hat, self.u_store.get(), self.Parameters)
            self.x_hat = self.state_function(
                self.x_hat, self.u_store.get(), self.Parameters)
            self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):
        """
        Update the state estimate with the measurement.
        Args:
            y (np.ndarray): Measurement vector for the update step.
        """
        self.C = self.measurement_function_jacobian(
            self.x_hat, self.Parameters)

        P_CT_matrix = self.P @ self.C.T

        S_matrix = self.C @ P_CT_matrix + self.R
        self.G = P_CT_matrix @ np.linalg.inv(S_matrix)
        self.x_hat = self.x_hat + self.G @ (y - self.measurement_function(
            self.x_hat, self.Parameters))

        self.P = (np.eye(self.A.shape[0]) - self.G @ self.C) @ self.P

    def get_x_hat_without_delay(self):
        """
        Get the current state estimate without considering input delays.
        Returns:
            np.ndarray: The current state estimate vector without delay.
        """
        if self.Number_of_Delay == 0:
            return self.x_hat
        else:
            x_hat = copy.deepcopy(self.x_hat)
            delay_index = self.u_store.delay_index + \
                self.Number_of_Delay - self._input_count

            for i in range(self._input_count):
                delay_index += 1
                if delay_index > self.Number_of_Delay:
                    delay_index = delay_index - self.Number_of_Delay - 1

                x_hat = self.state_function(
                    x_hat, self.u_store.get_by_index(delay_index), self.Parameters)

            return x_hat


class UnscentedKalmanFilter_Basic(KalmanFilterCommon):
    """
    UnscentedKalmanFilter_Basic implements a basic unscented Kalman filter for nonlinear systems.
    It uses nonlinear state and measurement functions, along with a method to calculate sigma points.
    Attributes:
        state_function (callable): Nonlinear state transition function.
        measurement_function (callable): Nonlinear measurement function.
        Q (np.ndarray): Process noise covariance matrix.
        R (np.ndarray): Measurement noise covariance matrix.
        x_hat (np.ndarray): Current state estimate.
        P (np.ndarray): Estimate error covariance.
        G (np.ndarray): Kalman gain.
        u_store (DelayedVectorObject): Object to handle delayed input vectors.
    """

    def __init__(self, state_function, measurement_function,
                 Q: np.ndarray, R: np.ndarray, Parameters=None,
                 Number_of_Delay=0, kappa=0.0):
        super().__init__(Number_of_Delay)
        self.state_function = state_function
        self.measurement_function = measurement_function

        self.Q = Q
        self.R = R

        self.STATE_SIZE = Q.shape[0]
        self.OUTPUT_SIZE = R.shape[0]

        self.x_hat = np.zeros((self.STATE_SIZE, 1))
        self.X_d = np.zeros((self.STATE_SIZE, 2 * self.STATE_SIZE + 1))

        self.P = np.eye(self.STATE_SIZE)
        self.G = None

        self.Parameters = Parameters
        self.u_store = None

        self.kappa = kappa
        self.sigma_point_weight = 0.0

        self.W = np.zeros((2 * self.STATE_SIZE + 1, 2 * self.STATE_SIZE + 1))

        self.calc_weights()

    def calc_weights(self):
        """
        Calculate the weights for the sigma points based on the kappa parameter.
        The weights are used to compute the mean and covariance of the sigma points.
        """
        lambda_weight = self.kappa

        self.W[0, 0] = self.kappa / \
            (self.STATE_SIZE + self.kappa)
        for i in range(1, 2 * self.STATE_SIZE + 1):
            self.W[i, i] = 1.0 / (2.0 * (self.STATE_SIZE + self.kappa))

        self.sigma_point_weight = math.sqrt(self.STATE_SIZE + lambda_weight)

    def calc_sigma_points(self, x: np.ndarray, P: np.ndarray):
        """
        Calculate the sigma points based on the current state estimate and covariance.
        Args:
            x (np.ndarray): Current state estimate vector.
            P (np.ndarray): Current estimate error covariance matrix.
        Returns:
            np.ndarray: Sigma points matrix, where each column is a sigma point.
        """
        sqrtP = np.linalg.cholesky(P, upper=True)
        Kai = np.zeros((self.STATE_SIZE, 2 * self.STATE_SIZE + 1))

        Kai[:, 0] = x.flatten()
        for i in range(self.STATE_SIZE):
            Kai[:, i + 1] = (x + self.sigma_point_weight *
                             (sqrtP[:, i]).reshape(-1, 1)).flatten()
            Kai[:, i + self.STATE_SIZE + 1] = \
                (x - self.sigma_point_weight *
                 (sqrtP[:, i]).reshape(-1, 1)).flatten()

        return Kai

    def predict(self, u: np.ndarray):
        """
        Predict the next state based on the current state and input.
        Args:
            u (np.ndarray): Input vector for the prediction step.
        """
        if self.u_store is None:
            self.u_store = DelayedVectorObject(
                u.shape[0], self.Number_of_Delay)

        self.u_store.push(u)

        if self._input_count < self.Number_of_Delay:
            self._input_count += 1
        else:
            Kai = self.calc_sigma_points(self.x_hat, self.P)

            for i in range(2 * self.STATE_SIZE + 1):
                Kai[:, i] = self.state_function(
                    Kai[:, i].reshape(-1, 1), u, self.Parameters).flatten()

            self.x_hat = np.zeros((self.STATE_SIZE, 1))
            for i in range(2 * self.STATE_SIZE + 1):
                self.x_hat += self.W[i, i] * Kai[:, i].reshape(-1, 1)

            for i in range(2 * self.STATE_SIZE + 1):
                self.X_d[:, i] = (Kai[:, i].reshape(-1, 1) -
                                  self.x_hat).flatten()

            self.P = self.X_d @ self.W @ self.X_d.T + self.Q

    def update(self, y):
        """
        Update the state estimate with the measurement.
        Args:
            y (np.ndarray): Measurement vector for the update step.
        """
        Kai = self.calc_sigma_points(self.x_hat, self.P)

        Y_d = np.zeros((self.OUTPUT_SIZE, 2 * self.STATE_SIZE + 1))
        for i in range(2 * self.STATE_SIZE + 1):
            Y_d[:, i] = self.measurement_function(
                Kai[:, i].reshape(-1, 1), self.Parameters).flatten()

        y_hat_m = np.zeros((self.OUTPUT_SIZE, 1))
        for i in range(2 * self.STATE_SIZE + 1):
            y_hat_m += self.W[i, i] * Y_d[:, i].reshape(-1, 1)

        for i in range(2 * self.STATE_SIZE + 1):
            Y_d[:, i] = (Y_d[:, i].reshape(-1, 1) - y_hat_m).flatten()

        P_yy = Y_d @ self.W @ Y_d.T
        P_xy = self.X_d @ self.W @ Y_d.T

        self.G = P_xy @ np.linalg.inv(P_yy + self.R)

        self.x_hat = self.x_hat + self.G @ (y - y_hat_m)
        self.P = self.P - self.G @ P_xy.T

    def get_x_hat_without_delay(self):
        """
        Get the current state estimate without considering input delays.
        Returns:
            np.ndarray: The current state estimate vector without delay.
        """
        if self.Number_of_Delay == 0:
            return self.x_hat
        else:
            x_hat = copy.deepcopy(self.x_hat)
            delay_index = self.u_store.delay_index + \
                self.Number_of_Delay - self._input_count

            for i in range(self._input_count):
                delay_index += 1
                if delay_index > self.Number_of_Delay:
                    delay_index = delay_index - self.Number_of_Delay - 1

                x_hat = self.state_function(
                    x_hat, self.u_store.get_by_index(delay_index), self.Parameters)

            return x_hat


class UnscentedKalmanFilter(UnscentedKalmanFilter_Basic):
    """
    UnscentedKalmanFilter implements an unscented Kalman filter for nonlinear systems with additional parameters for tuning.
    It extends the basic unscented Kalman filter with parameters for kappa, alpha, and beta.
    Attributes:
        kappa (float): Scaling parameter for the sigma points.
        alpha (float): Parameter for the spread of the sigma points.
        beta (float): Parameter for the optimality of the filter.
    """

    def __init__(self, state_function, measurement_function,
                 Q: np.ndarray, R: np.ndarray, Parameters=None,
                 Number_of_Delay=0, kappa=0.0, alpha=0.5, beta=2.0):

        self.kappa = 0.0
        if kappa == 0.0:
            self.kappa = 3 - Q.shape[0]
        else:
            self.kappa = kappa

        self.alpha = alpha
        self.beta = beta
        self.sigma_point_weight = 0.0

        self.w_m = 0.0

        super().__init__(state_function, measurement_function,
                         Q, R, Parameters, Number_of_Delay, self.kappa)

    def calc_weights(self):
        """
        Calculate the weights for the sigma points based on the alpha, beta, and kappa parameters.
        The weights are used to compute the mean and covariance of the sigma points.
        """
        lambda_weight = self.alpha * self.alpha * \
            (self.STATE_SIZE + self.kappa) - self.STATE_SIZE

        self.w_m = lambda_weight / \
            (self.STATE_SIZE + lambda_weight)
        self.W[0, 0] = self.w_m + 1.0 - self.alpha * self.alpha + self.beta
        for i in range(1, 2 * self.STATE_SIZE + 1):
            self.W[i, i] = 1.0 / (2.0 * (self.STATE_SIZE + lambda_weight))

        self.sigma_point_weight = math.sqrt(self.STATE_SIZE + lambda_weight)

    def predict(self, u):
        """
        Predict the next state based on the current state and input.
        Args:
            u (np.ndarray): Input vector for the prediction step.
        """
        if self.u_store is None:
            self.u_store = DelayedVectorObject(
                u.shape[0], self.Number_of_Delay)

        self.u_store.push(u)

        if self._input_count < self.Number_of_Delay:
            self._input_count += 1
        else:
            Kai = self.calc_sigma_points(self.x_hat, self.P)

            for i in range(2 * self.STATE_SIZE + 1):
                Kai[:, i] = self.state_function(
                    Kai[:, i].reshape(-1, 1), u, self.Parameters).flatten()

            self.x_hat = self.w_m * Kai[:, 0].reshape(-1, 1)
            for i in range(1, 2 * self.STATE_SIZE + 1):
                self.x_hat += self.W[i, i] * Kai[:, i].reshape(-1, 1)

            for i in range(2 * self.STATE_SIZE + 1):
                self.X_d[:, i] = (Kai[:, i].reshape(-1, 1) -
                                  self.x_hat).flatten()

            self.P = self.X_d @ self.W @ self.X_d.T + self.Q

    def update(self, y):
        """
        Update the state estimate with the measurement.
        Args:
            y (np.ndarray): Measurement vector for the update step.
        """
        Kai = self.calc_sigma_points(self.x_hat, self.P)

        Y_d = np.zeros((self.OUTPUT_SIZE, 2 * self.STATE_SIZE + 1))
        for i in range(2 * self.STATE_SIZE + 1):
            Y_d[:, i] = self.measurement_function(
                Kai[:, i].reshape(-1, 1), self.Parameters).flatten()

        y_hat_m = self.w_m * Y_d[:, 0].reshape(-1, 1)
        for i in range(1, 2 * self.STATE_SIZE + 1):
            y_hat_m += self.W[i, i] * Y_d[:, i].reshape(-1, 1)

        for i in range(2 * self.STATE_SIZE + 1):
            Y_d[:, i] = (Y_d[:, i].reshape(-1, 1) - y_hat_m).flatten()

        P_yy = Y_d @ self.W @ Y_d.T
        P_xy = self.X_d @ self.W @ Y_d.T

        self.G = P_xy @ np.linalg.inv(P_yy + self.R)

        self.x_hat = self.x_hat + self.G @ (y - y_hat_m)
        self.P = self.P - self.G @ P_xy.T
