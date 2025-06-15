"""
This module implements a discrete-time PID (Proportional-Integral-Derivative) controller with anti-windup and derivative filtering features.
It provides a class for configuring and updating a PID controller in a digital control system, supporting output saturation and back-calculation for integral anti-windup.
"""
import math
import numpy as np

MINIMUM_OUTPUT_DEFAULT = -np.inf
MAXIMUM_OUTPUT_DEFAULT = np.inf


def saturation(value, min_value, max_value):
    return max(min(value, max_value), min_value)


class DiscretePID_Controller:
    """
    A class for implementing a discrete-time PID controller with anti-windup and derivative filtering.
    This class allows configuring PID gains, delta time, and output limits, and provides methods for updating the controller based on error inputs.
    Attributes:
        delta_time (float): The time interval between updates.
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        N (float): Filter coefficient for derivative term.
        Kb (float): Back calculation coefficient for anti-windup.
        minimum_output (float): Minimum output limit.
        maximum_output (float): Maximum output limit.
    """

    def __init__(self, delta_time=0.0, Kp=0.0, Ki=0.0, Kd=0.0, N=100.0, Kb=0.0,
                 minimum_output=MINIMUM_OUTPUT_DEFAULT, maximum_output=MAXIMUM_OUTPUT_DEFAULT):
        # Public variables
        self.delta_time = delta_time
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.N = N  # Filter coefficient
        self.Kb = Kb  # Back calculation coefficient
        self.minimum_output = minimum_output
        self.maximum_output = maximum_output

        # Private variables
        self._integration_store = 0.0
        self._differentiation_store = 0.0
        self._PID_output = 0.0
        self._saturated_PID_output = 0.0

        if isinstance(self.Kp, float) and \
                isinstance(self.Ki, float) and \
                isinstance(self.Kd, float):

            self.data_type = np.float64.__name__
        else:
            raise TypeError("controller gains must be a float")

    def update(self, error):
        """
        Update the PID controller with a new error value.
        Args:
            error (float): The current error value (setpoint - process variable).
        Returns:
            float: The saturated PID output value.
        """
        self._PID_output = (self._calculate_P_term(error) +
                            self._calculate_I_term(error) +
                            self._calculate_D_term(error))

        self._saturated_PID_output = saturation(
            self._PID_output, self.minimum_output, self.maximum_output)

        return self._saturated_PID_output

    def reset(self):
        """
        Reset the PID controller's internal state.
        This method clears the integration and differentiation stores,
        effectively resetting the controller to its initial state.
        """
        self._integration_store = 0.0
        self._differentiation_store = 0.0

    def get_integration_store(self):
        """
        Get the current value of the integration store.
        Returns:
            float: The current value of the integration store.
        """
        return self._integration_store

    def get_differentiation_store(self):
        """
        Get the current value of the differentiation store.
        Returns:
            float: The current value of the differentiation store.
        """
        return self._differentiation_store

    def _back_calculation_for_I_term(self, error):
        """
        Calculate the back-calculation term for the integral part of the PID controller.
        This term is used to prevent integral windup by adjusting the integral term based on the difference
        between the saturated PID output and the current PID output.
        Args:
            error (float): The current error value.
        Returns:
            float: The back-calculation term for the integral part.
        """
        return (self.Ki * error +
                self.Kb * (self._saturated_PID_output - self._PID_output))

    def _calculate_P_term(self, error):
        """
        Calculate the proportional term of the PID controller.
        Args:
            error (float): The current error value.
        Returns:
            float: The proportional term of the PID controller.
        """
        return self.Kp * error

    def _calculate_I_term(self, error):
        """
        Calculate the integral term of the PID controller.
        This method uses a back-calculation approach to prevent integral windup.
        Args:
            error (float): The current error value.
        Returns:
            float: The integral term of the PID controller.
        """
        self._integration_store += (self._back_calculation_for_I_term(error) *
                                    self.delta_time)
        return self._integration_store

    def _calculate_D_term(self, error):
        """
        Calculate the derivative term of the PID controller.
        This method uses a simple differentiation approach, which may not be ideal for all applications.
        Args:
            error (float): The current error value.
        Returns:
            float: The derivative term of the PID controller.
        """
        # Incomplete differentiator
        output = self.N * ((self.Kd * error) - self._differentiation_store)

        self._differentiation_store += output * self.delta_time

        return output
