# -*- coding = utf-8 -*-
import numpy as np

from tqdm import tqdm


class KNNClassifier(object):
    """The k-nearest neighbors classifier algorithm, implemented from scratch."""
    def __init__(self, k = 1, num_classes = None):
        # Initialize class values.
        self.k = k
        self._is_fitted = False
        self._num_classes = num_classes

    def __str__(self):
        return f"<k-nearest neighbors with k={self.k}>"

    @staticmethod
    def _euclidean_distance(x1, x2):
        # Calculates Euclidean distance between two data points.
        return np.linalg.norm(x1 - x2)

    @staticmethod
    def _resolve_data(X, y):
        """Resolves and validates input X and y data."""
        if y is None:
            try:
                # Check if `X` is a tuple of two elements.
                X, y = X
            except:
                if isinstance(X, (list, tuple, np.ndarray)):
                    # Check if `X` is a list of pairs of data.
                    if len(X[0]) == 2 and len(X[1]) == 2:
                        X, y = [i[0] for i in X], [i[1] for i in y]
                else:
                    raise ValueError(
                        "Expected two arrays 'X' and 'y', a list "
                        "of two arrays 'X', or a list of data pairs.")

        # Return the data.
        return X, y

    @staticmethod
    def _to_list(item):
        """Converts an item from an ndarray to a list."""
        if isinstance(item, (list, tuple)):
            return item
        if isinstance(item, np.ndarray):
            item = np.split(item, len(item),axis = 0)
        return item

    def _check_assign_classes(self, y):
        """Checks the provided class labels."""
        classes = np.unique(y)
        if not np.issubdtype(classes.dtype, np.integer):
            raise TypeError(
                f"Expected integer labels, instead got {classes.dtype}")
        if self._num_classes is not None and len(classes) != self._num_classes:
            raise ValueError(
                f"Expected {self._num_classes} classes, got {len(classes)}.")
        return classes

    def _append_fit(self, X, y):
        """Adds more data to the classifier."""
        prev_X, prev_y = self._to_list(self.X), self._to_list(self.y)
        X, y = self._to_list(X), self._to_list(y)
        prev_X.append(X), prev_y.append(y)
        self.X = prev_X, self.y = prev_y

    def fit(self, X, y = None):
        """Fits the classifier to the provided input data."""
        # Resolve and assign the data to the classifier.
        X, y = self._resolve_data(X, y)
        if not self._is_fitted:
            self._assign_data(X, y)
        else:
            self._append_fit(X, y)

        # Check and/or determine the class labels.
        classes = np.unique(y)
        if not np.issubdtype(classes.dtype, np.integer):
            raise TypeError(
                f"Expected integer labels, instead got {classes.dtype}")
        if self._num_classes is not None and len(classes) != self._num_classes:
            raise ValueError(
                f"Expected {self._num_classes} classes, got {len(classes)}.")
        self._num_classes = len(classes)

        self._is_fitted = True

    def _assign_data(self, X, y):
        """Assigns a set of data to the classifier."""
        self.X, self.y = X, y

    def predict(self, value, verbose = False):
        """Predicts the class of an input based on the data."""
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                value = [value]
        value = self._to_list(value)

        # Create a verbose progress bar if requested to.
        if verbose:
            value = tqdm(value, desc = "Running Inference")

        # Calculate the euclidean distances.
        predicted = []
        for data in value:
            # Calculate the store the distances and their corresponding labels.
            dists, labels = [], []
            for X_, y_ in zip(self.X, self.y):
                dists.append(self._euclidean_distance(X_, data))
                labels.append(y_)
            dists, labels = np.array(dists), np.array(labels)

            # Find the smallest distances and their labels.
            permutation = dists.argsort()
            most_common = labels[permutation][:self.k]
            if len(most_common) == 1:
                predicted.append(most_common)
            else:
                values, counts = np.unique(most_common, return_counts = True)
                predicted.append(values[np.argmax(counts)])

        # Return the predictions.
        return predicted[0] if len(value) == 1 else predicted











