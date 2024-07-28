class Node:
    """
    A class to represent a node in the decision tree.
    Each node stores the feature and threshold used for splitting,
    the left and right child nodes, and a value if it's a leaf node.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        # A node is a leaf node if it has a value
        return self.value is not None
