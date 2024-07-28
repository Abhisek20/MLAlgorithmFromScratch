import numpy as np
from utils.Node import Node


class DecisionTreeRegressor():
    """
    A decision tree regressor class that fits to the data and makes predictions.
    """

    def __init__(self, min_samples_split=2, min_samples_leaf=2, min_variance_reduction=0.01, max_depth=100, n_features=None, ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.min_samples_leaf = min_samples_leaf
        self.min_variance_reduction = min_variance_reduction
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Fit the model to the data
        self.n_features = X.shape[1] if self.n_features is None else min(
            X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        # Predict values for the input data
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """
        Recursively grow the decision tree by finding the best splits.
        """
        n_samples, n_feats = X.shape

        # Stop if max depth is reached or the number of samples is less than min_samples_split
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return Node(value=np.mean(y))  # Leaf node with the mean value

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)

        if best_feature is None:
            return Node(value=np.mean(y))  # Leaf node if no valid split found

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)

        # Ensure both child nodes have at least min_samples_split samples
        if len(left_idxs) < self.min_samples_split or len(right_idxs) < self.min_samples_split:
            # Leaf node if either child node is too small
            return Node(value=np.mean(y))

        # Recursively grow the left and right subtrees
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X: np.ndarray, y: np.ndarray, feat_idxs: np.ndarray):
        """
        Find the best feature and threshold to split on, based on variance reduction.
        """
        best_variance = -np.inf
        split_idx = split_threshold = None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                var_gain = self._variance_reduction(y, X_column, thr)
                if var_gain > best_variance:
                    best_variance = var_gain
                    split_idx = feat_idx
                    split_threshold = thr

        # Consider a split only if it reduces the variance by at least min_variance_reduction
        if best_variance >= self.min_variance_reduction:
            return split_idx, split_threshold
        else:
            return None, None

    def _variance_reduction(self, y: np.ndarray, X_column: np.ndarray, threshold: float):
        """
        Calculate the variance reduction from splitting on a specific feature threshold.
        """
        parent_variance = np.var(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return -np.inf

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = np.var(y[left_idxs]), np.var(y[right_idxs])
        child_variance = (n_l / n) * e_l + (n_r / n) * e_r

        variance_reduction = parent_variance - child_variance
        return variance_reduction

    def _split(self, X_column: np.ndarray, split_thresh: float):
        """
        Split the data into left and right branches based on the threshold.
        """
        left_idxs = np.where(X_column <= split_thresh)[0]
        right_idxs = np.where(X_column > split_thresh)[0]
        return left_idxs, right_idxs

    def _traverse_tree(self, x: np.ndarray, node: Node):
        """
        Traverse the tree to make a prediction for a single sample.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
