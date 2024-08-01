import numpy as np
from enum import Enum
from numba import jit


class SolverMethod(Enum):
    """Enumeration for solver methods in linear regression."""
    NORMAL_EQUATION = 0  # Method using normal equations
    SGD = 1  # Method using stochastic gradient descent


class LinearRegression:
    """Linear Regression model class implementing both normal equation and SGD methods."""

    def __init__(self, method=SolverMethod.NORMAL_EQUATION, epochs: int = 100, learning_rate: float = 0.01, batch_size: int = 32) -> None:
        """
        Initialize the Linear Regression model.

        Parameters:
        method (SolverMethod): Method to use for fitting the model.
        epochs (int): Number of epochs for SGD.
        learning_rate (float): Learning rate for SGD.
        batch_size (int): Batch size for SGD.
        """
        self.method = method
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def fit(self, X: np.array, y: np.array):
        """
        Fit the Linear Regression model to the training data.

        Parameters:
        X (np.array): Feature matrix.
        y (np.array): Target vector.
        """
        self.num_rows, self.num_cols = X.shape
        # Add a bias term (column of ones) to the feature matrix
        self.X_b = np.concatenate([np.ones((self.num_rows, 1)), X], axis=1)
        if self.method == SolverMethod.NORMAL_EQUATION:
            self.intercept, self.coefs = self._fit_normal_equation(self.X_b, y)
        elif self.method == SolverMethod.SGD:
            param_arr = np.random.randn(self.num_cols + 1, 1)
            self.intercept, self.coefs = fit_sgd(
                self.X_b, y, param_arr, self.learning_rate, self.epochs, self.batch_size)

    def predict(self, X):
        """
        Predict the target values for the given input data.

        Parameters:
        X (np.array): Feature matrix for prediction.

        Returns:
        np.array: Predicted target values.
        """
        # Compute the predicted values using the learned coefficients
        return self.intercept + self.coefs @ X.T

    def _fit_normal_equation(self, X, y):
        """
        Fit the model using the normal equation method.

        Parameters:
        X (np.array): Feature matrix with bias term.
        y (np.array): Target vector.

        Returns:
        float, np.array: Intercept and coefficients of the model.
        """
        # Compute the optimal coefficients using the normal equation
        param_arr = np.linalg.inv(X.T @ X) @ X.T @ y
        intercept, coefs = param_arr[0], param_arr[1:]
        return intercept, coefs


@jit(nopython=True, fastmath=True)
def calculate_batch_gradients(X_batch: np.array, y_batch: np.array, param_arr: np.array):
    """
    Calculate the gradients for a batch of data.

    Parameters:
    X_batch (np.array): Feature matrix batch.
    y_batch (np.array): Target vector batch.
    param_arr (np.array): Current coefficients.

    Returns:
    np.array: Gradients for the coefficients.
    """
    sample_size = X_batch.shape[0]

    # first order derivative of mse loss  :
    #  2*(y_hat - y)@X/sample_size = 2*(X_T@param_arr - y)@X/sample_size = 2*error@X/sample_size

    # (bs x num_cols) x (num_cols x 1 ) = (bs x 1)
    prediction = X_batch @ param_arr

    error = prediction - y_batch
    # (bs x num_cols).T x (bs x 1)  = (num_cols x 1)
    gradients = 2 * (X_batch.T @ error) / sample_size
    return gradients


@jit(nopython=True, fastmath=True)
def fit_sgd(X: np.array, y: np.array, param_arr: np.array, learning_rate: float, epochs: int, batch_size: int):
    """
    Fit the model using stochastic gradient descent (SGD) with Numba optimization.

    Parameters:
    X (np.array): Feature matrix with bias term.
    y (np.array): Target vector.
    param_arr (np.array): Initial coefficients.
    learning_rate (float): Learning rate for SGD.
    epochs (int): Number of epochs.
    batch_size (int): Size of each batch.

    Returns:
    np.array: Updated coefficients.
    """
    num_rows = X.shape[0]
    for _ in range(epochs):
        rand_idxs = np.random.permutation(num_rows)
        X_shuffled = X[rand_idxs]
        y_shuffled = y[rand_idxs]
        for idx in range(0, num_rows, batch_size):
            X_batch = X_shuffled[idx:idx + batch_size]
            y_batch = y_shuffled[idx:idx + batch_size]
            gradients = calculate_batch_gradients(X_batch, y_batch, param_arr)
            param_arr -= learning_rate * gradients

    intercept, coefs = param_arr[0], param_arr[1:]
    return intercept, coefs
