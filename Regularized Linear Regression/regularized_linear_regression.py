
import numpy as np
from enum import Enum
from numba import jit


class RegularizedLinearRegression:
    """Linear Regression model class implementing both normal equation and SGD methods."""

    def __init__(self,
                 regularization="l1",
                 alpha = 0.01,
                 epochs: int = 100,
                 learning_rate: float = 0.01,
                 batch_size: int = 32) -> None:
        """
        Initialize the Linear Regression model.

        Parameters:
        regularization (str) : Type of regularization.
        alpha (float) : Regularization strength. 
        epochs (int): Number of epochs for SGD.
        learning_rate (float): Learning rate for SGD.
        batch_size (int): Batch size for SGD.
        """
        self.regularization = regularization
        self.alpha = alpha
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def fit(self, X: np.array, y: np.array,):
        """
        Fit the Linear Regression model to the training data.

        Parameters:
        X (np.array): Feature matrix.
        y (np.array): Target vector.
        """
        self.num_rows, self.num_cols = X.shape
        # Add a bias term (column of ones) to the feature matrix
        self.X_b = np.concatenate([np.ones((self.num_rows, 1)), X], axis=1)

        param_arr = np.random.randn(self.num_cols + 1, 1)
        self.intercept, self.coefs = fit_sgd(
            self.X_b, y, param_arr, self.learning_rate, self.epochs, self.batch_size, self.alpha, self.regularization)

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
        return X_b @ np.concatenate([[self.intercept], self.coefs],
                                    axis=0)


@jit(nopython=True, fastmath=True)
def fit_sgd(X: np.array, y: np.array, param_arr: np.array, learning_rate: float, epochs: int, batch_size: int, alpha: float, regularization: str):
    """
    Fit the model using stochastic gradient descent (SGD) with Numba optimization.

    Parameters:
    X (np.array): Feature matrix with bias term.
    y (np.array): Target vector.
    param_arr (np.array): Initial coefficients.
    learning_rate (float): Learning rate for SGD.
    epochs (int): Number of epochs.
    batch_size (int): Size of each batch.
    alpha (float) : Regularization strength.
    regularization (str) : Type of regularization.

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
            # for regularization the loss update has an extra paramater which adds more penalty
            if regularization == "l1":
                param_arr -= (learning_rate * gradients +
                              alpha*np.sign(param_arr))
            elif regularization == "l2":
                param_arr -= (learning_rate * gradients + alpha*param_arr)
            else:
                param_arr -= learning_rate * gradients

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
