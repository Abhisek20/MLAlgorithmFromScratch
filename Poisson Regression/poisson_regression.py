import numpy as np
from enum import Enum
from numba import jit


class PoissonRegression:
    """Poisson(GLM) Regression model class implementing both normal equation and SGD methods with gradient clipping."""

    def __init__(
        self,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        tolerance: float = 1e-6,
    ) -> None:
        """
        Initialize the Poisson(GLM) Regression model.

        Parameters:
        epochs (int): Number of epochs for SGD.
        learning_rate (float): Learning rate for SGD.
        batch_size (int): Batch size for SGD.
        tolerance (float) : Minimum weight update.
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tolerance = tolerance

    def fit(self, X: np.array, y: np.array):
        """
        Fit the Poisson(GLM) Regression model to the training data.

        Parameters:
        X (np.array): Feature matrix.
        y (np.array): Target vector.
        """
        self.num_rows, self.num_cols = X.shape
        # Add a bias term (column of ones) to the feature matrix
        self.X_b = np.concatenate([np.ones((self.num_rows, 1)), X], axis=1)

        param_arr = np.random.randn(self.num_cols + 1, 1)
        self.intercept, self.coefs = fit_sgd(
            self.X_b,
            y,
            param_arr,
            self.learning_rate,
            self.epochs,
            self.batch_size,
            self.tolerance,
        )

    def predict(self, X):
        """
        Predict the target values for the given input data.

        Parameters:
        X (np.array): Feature matrix for prediction.

        Returns:
        np.array: Predicted target values.
        """
        # Compute the predicted values using the learned coefficients
        # Add a bias term to the feature matrix
        X_b = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return get_lambda(X_b, np.concatenate([[self.intercept], self.coefs], axis=0))


@jit(nopython=True, fastmath=True)
def get_lambda(x: np.array, betas: np.array) -> np.array:
    # Clip the input to the exponential function to prevent overflow
    return np.exp(np.clip(x @ betas, 0, 1e6))


@jit(nopython=True, fastmath=True)
def fit_sgd(
    X: np.array,
    y: np.array,
    param_arr: np.array,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    tolerance: float,
):
    num_rows = X.shape[0]

    for epoch in range(epochs):
        rand_idxs = np.random.permutation(num_rows)
        X_shuffled = X[rand_idxs]
        y_shuffled = y[rand_idxs]

        for idx in range(0, num_rows, batch_size):
            X_batch = X_shuffled[idx : idx + batch_size]
            y_batch = y_shuffled[idx : idx + batch_size]

            # Calculate gradients
            gradients = calculate_batch_gradients(X_batch, y_batch, param_arr)

            # Clip gradients to prevent large updates
            gradients = np.clip(gradients, -1, 1)

            # Calculate parameter update
            param_update = learning_rate * gradients

            # Check for NaN or Inf in the update
            if np.any(np.isnan(param_update)) or np.any(np.isinf(param_update)):
                print(f"NaN or Inf detected in parameter update at epoch {epoch}, batch {idx // batch_size}")
                return param_arr[0], param_arr[1:]

            # Update the parameters
            param_arr += param_update

            # Check if the update is smaller than tolerance
            if np.linalg.norm(param_update) < tolerance:
                print(f"Stopping early at epoch {epoch}, batch {idx // batch_size}")
                return param_arr[0], param_arr[1:]

    return param_arr[0], param_arr[1:]



@jit(nopython=True, fastmath=True)
def calculate_batch_gradients(
    X_batch: np.array, y_batch: np.array, param_arr: np.array
):
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

    # (bs x num_cols) x (num_cols x 1 ) = (bs x 1)
    prediction = get_lambda(X_batch, param_arr)

    # (bs x 1) - (bs x 1) = (bs x 1)
    error = y_batch - prediction

    # (bs x num_cols).T x (bs x 1)  = (num_cols x 1)
    gradients = (X_batch.T @ error) / sample_size
    return gradients



