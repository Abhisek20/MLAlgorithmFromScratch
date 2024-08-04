import numpy as np
from numba import jit


class KNNRegressor():
    """
    K-Nearest Neighbors Regressor.

    Parameters:
    k (int): Number of nearest neighbors to consider.
    """

    def __init__(self, k=5) -> None:
        self.k = k

    def fit(self, X: np.array, y: np.array):
        """
        Fit the KNN regressor model.

        Parameters:
        X (np.array): Training data features.
        y (np.array): Training data targets.
        """
        self.rows_train, _ = X.shape
        self.X_train, self.y_train = X, y

    def predict(self, X: np.array) -> np.array:
        """
        Predict the target values for the given test data.

        Parameters:
        X (np.array): Test data features.

        Returns:
        np.array: Predicted target values.
        """
        self.rows_test, _ = X.shape
        y_pred = np.zeros(self.rows_test)

        for idx in range(self.rows_test):
            # Calculate the target values of the k nearest neighbors
            neighbour_targets = _calculate_neighbour_targets(
                self.k, X[idx], self.X_train, self.y_train, self.rows_train)

            # Predict the target value as the mean of the nearest neighbors' targets
            y_pred[idx] = np.mean(neighbour_targets)

        return y_pred


@jit(nopython=True, fastmath=True)
def _calculate_neighbour_targets(k: int, X_idx: np.array, X_train: np.array, y_train: np.array, rows_train: int) -> np.array:
    """
    Calculate the target values of the k nearest neighbors.

    Parameters:
    k (int): Number of nearest neighbors to consider.
    X_idx (np.array): Single test data point.
    X_train (np.array): Training data features.
    y_train (np.array): Training data targets.
    rows_train (int): Number of training data points.

    Returns:
    np.array: Target values of the k nearest neighbors.
    """
    euc_dist = np.zeros(rows_train)

    for idx in range(rows_train):
        # Calculate the Euclidean distance between the test point and each training point
        dist = _calculate_minkowski_distance(X_idx, X_train[idx])
        euc_dist[idx] = dist

    # Get the indices that would sort the distances array
    sorted_idxs = euc_dist.argsort()

    # Sort the training targets based on the sorted distances
    y_train_sorted = y_train[sorted_idxs]

    return y_train_sorted[:k]


@jit(nopython=True, fastmath=True, cache=True)
def _calculate_minkowski_distance(X_test: np.array, X_train: np.array) -> float:
    """
    Calculate the Minkowski distance (Euclidean distance) between two points.

    Parameters:
    X_test (np.array): Test data point.
    X_train (np.array): Training data point.

    Returns:
    float: Euclidean distance between the two points.
    """
    # Euclidean distance = √Σ(x_i - y_i)^2, for i=1,2,3,...<# of columns>
    return np.sqrt(np.sum(np.square(X_test - X_train)))
