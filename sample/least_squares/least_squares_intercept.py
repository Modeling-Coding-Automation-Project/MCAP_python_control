"""Demonstrate least-squares fitting with and without an intercept."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python_control.least_squares import LeastSquares


def main():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([[2.0], [3.0], [4.0]])

    with_intercept = LeastSquares(X)
    without_intercept = LeastSquares(X, fit_intercept=False)
    with_intercept.fit(X, y)
    without_intercept.fit(X, y)

    np.testing.assert_allclose(with_intercept.get_weights(), [[1.0], [1.0]])
    np.testing.assert_allclose(without_intercept.get_weights(), [[10.0 / 7.0]])
    np.testing.assert_allclose(with_intercept.predict(X), y)
    np.testing.assert_allclose(without_intercept.predict(X), X * (10.0 / 7.0))
    np.testing.assert_allclose(
        without_intercept.predict(np.zeros_like(X)), 0.0)

    print("with intercept weights:", with_intercept.get_weights().ravel())
    print("without intercept weights:", without_intercept.get_weights().ravel())


if __name__ == "__main__":
    main()
