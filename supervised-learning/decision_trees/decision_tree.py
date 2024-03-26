import numpy as np
from collections import Counter

class Node:
    """Node data structure for decision tree algorithm"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value
    
    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    """Decision tree algorithm applied with Entropy and Information Gain"""
    def __init__(self, max_depth=100):
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y, 0)
    
    def _grow_tree(self, X:np.ndarray, y:np.ndarray, depth):
        n_feats = X.shape[1]
        n_labels = len(np.unique(y))

        # stopping the recursion to return the leaf node
        if (depth >= self.max_depth or n_labels == 1):
            leaf_value = self._best_fitting_label(y)
            return Node(value=leaf_value)
        

        # no leaf node. so create parent nodes with children
        feature_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        
        # create children
        best_threshold, best_feature = self._best_split(X, y, feature_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature, best_threshold, left, right)


    def _best_fitting_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value


    def _best_split(self, X, y, feature_idxs):
        best_gain = -1 
        split_idx, split_threshold = None, None

        for feat_idx in feature_idxs:
            x_col = X[:, feat_idx]
            thresholds = np.unique(x_col)

            for thr in thresholds:
                gain = self._information_gain(y, x_col, thr)

                if gain > best_gain:
                    split_idx = feat_idx
                    split_threshold = thr
                    best_gain = gain
        
        return split_threshold, split_idx


    
    def _information_gain(self, y, x_col, threshold):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(x_col, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(right_idxs)

        info_gain = parent_entropy - ((n_l / n) * e_l  +  (n_r / n) * e_r)
        return info_gain



    def _entropy(self, y):
        hist = np.bincount(y)
        probs = hist / len(y)
        return -np.sum([p * np.log(p) for p in probs if p > 0])
    

    def _split(self, x_col, threshold):
        left_idxs = np.argwhere(x_col <= threshold).flatten()
        right_idxs = np.argwhere(x_col > threshold).flatten()
        return left_idxs, right_idxs

    def predict(self, X:np.ndarray):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node:Node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    