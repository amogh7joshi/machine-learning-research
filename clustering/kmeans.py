#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import time

import numpy as np
import matplotlib.pyplot as plt

# Configure plot style.
from matplotlib import style
style.use('ggplot')

# Get colors for clusters.
import matplotlib.colors as mcolors

# Convenience for creating training data.
from sklearn.datasets import make_blobs

class KMeans(object):
   """The K-Means clustering algorithm, implemented from scratch (for 2-D clustering)."""
   def __init__(self, K = None):
      # Initialize class values.
      self.K = K
      self.centroids = {}
      self.classes = {}
      self.data = None

   def __str__(self):
      # The centroids define the class.
      return self.centroids

   def __len__(self):
      if self.K is None: # If class has not been fit (and K is uninitialized).
         raise NotImplementedError("You have not provided a value for K, so you need to fit the "
                                   "algorithm to training data first before trying to find its length.")
      return self.K

   @staticmethod
   def _assert_dimensions(*arrays):
      """Ensures that provided data is two-dimensional, for compatibility"""
      for array in arrays:
         if isinstance(array, np.ndarray):
            if not len(array.shape) == 2:
               raise ValueError(f"The provided data should be two-dimensional, "
                                f"got {len(array.shape)} dimension(s).")
         elif isinstance(array, (list, tuple, set)):
            if not len(np.array(array).shape) == 2:
               raise ValueError(f"The provided data should be two-dimensional, "
                                f"got {len(np.array(array).shape)} dimension(s).")

   def create_blob_dataset(self, samples, features = None, state = None):
      """A wrapper for the scikit-learn make_blobs method for custom cluster datasets."""
      # Feature number will usually be the class dataset.
      if features is None:
         features = self.K

      # Set the class data.
      setattr(self, "dataset", make_blobs(n_samples = samples, n_features = features,
                                          centers = features, random_state = state)[0])
      return self

   @staticmethod
   def euclidean_distance(p1, p2):
      """Returns the euclidean distance between two points."""
      if isinstance(p2, tuple):
         # Issues with dictionary comprehensions.
         p2 = p2[1]
      return np.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

   @staticmethod
   def _initialize_centroids(X, K):
      """Initializes the centroids to the first N values in the dataset."""
      centroids = {}
      for i in range(K):
         centroids[i] = X[i]
      return centroids

   @staticmethod
   def _is_optimized(previous_centroids, current_centroids, tolerance):
      """Compares previous and current centroids to determine if K-Means has reached its optimization objective."""
      for centroid in current_centroids:
         # Select previous and current centroids.
         previous = previous_centroids[centroid]
         current = current_centroids[centroid]

         if np.sum((current - previous) / current * 100) > tolerance:
            # If the difference is greater than allowed by class tolerance.
            return False

      # Otherwise, algorithm has reached its optimization objective.
      return True

   def _select_classifications(self, X, K):
      """The internal fitting algorithm, called for each individual iteration over the training data."""
      # Create a dictionary of classifications.
      classifications = {}
      for i in range(K):
         classifications[i] = []

      # Iterate over training data.
      for indx, point in enumerate(X):
         # Determine list of euclidean distances.
         euclidean_distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids.items()]

         # Create data classifications to update with.
         classifications[euclidean_distances.index(min(euclidean_distances))].append(point)

      # Return the calculated classifications.
      return classifications

   def _fit_with_k_value(self, X, K, epochs, tolerance, verbose, visualize):
      """Fit the data with a pre-determined value for K."""
      self.centroids = self._initialize_centroids(X, K)

      # Determine whether to run a single time (for internal operations) or for provided epochs.
      for epoch in range(epochs):
         # Calculate the data classifications.
         self.classes = self._select_classifications(X, K)

         # Save the previous centroids.
         previous_centroids = self.centroids.copy()

         # Average the cluster data points to re-calculate centroids.
         for classification in self.classes:
            self.centroids[classification] = np.average(self.classes[classification], axis = 0)

         # Print centroids if verbosity is enabled.
         if verbose:
            if (epoch + 1) % verbose == 0:
               print(f"Epoch: {epoch}\t Current Centroids: {self.centroids}")

         # Determine whether class is optimized.
         if self._is_optimized(previous_centroids, self.centroids, tolerance):
            break

         # Display current centroid state after each epoch.
         if visualize:
            self.plot()
            time.sleep(2)

   def _iter_fit(self, X, epochs, tolerance, verbose, visualize):
      """Fit the data, by iterating over different values for K and picking the best."""
      raise NotImplementedError("The _iter_fit method has not been implemented yet. ")

   def fit(self, X = None, epochs = 100, tolerance = 0.001, verbose = True, visualize = True):
      """Fit the data using the K-Means algorithm."""
      # If no dataset is provided, then there may be an already-set class dataset.
      if X is None:
         if hasattr(self, "dataset"):
            X = self.dataset

      # Process verbosity, if set to True (prints out every 2% of training).
      if verbose:
         verbose = epochs // 50

      # Assign data to class (for future debugging, etc.).
      self.data = X

      # Fit data.
      if self.K is None: # If the number of clusters is not provided, then iterate over different values.
         self._iter_fit(X, epochs, tolerance, verbose, visualize)
      else: # Otherwise, if the number of clusters is provided, then automatically use that.
         self._fit_with_k_value(X, self.K, epochs, tolerance, verbose, visualize)

      # Return self (for stacked method calls).
      return self

   def predict(self, X, visualize = True):
      """Predicts the class label of a provided data point."""
      distances = []
      for centroid in self.centroids:
         # Calculate distance between each centroid, and find the minimum.
         distances.append(self.euclidean_distance(X, self.centroids[centroid]))

      # Find the class with the least distance between centroid/data.
      class_label = np.argmin(distances)

      # Display the data piece next to the rest of the centroids.
      if visualize:
         # Plot the centroids.
         for centroid in self.centroids:
            plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1], s = 130, marker = "x",
                        linewidths = 5, facecolor = 'black', edgecolor = 'black')

         # Plot the data point.
         plt.scatter(X[0], X[1], color = 'blue')

         # Display the plot.
         plt.show()

      # Return the label.
      return class_label

   def plot_scatter_data(self, X, y = None):
      """Creates a scatter plot of provided data points."""
      self._assert_dimensions(X)

      # Generate list of colors.
      _colors = mcolors.BASE_COLORS
      _colors.update(mcolors.TABLEAU_COLORS)
      _colors.update(mcolors.CSS4_COLORS)
      colors = _colors.copy()

      # Remove white entirely, that is reserved for the centroids.
      try:
         del _colors['w'], colors['w']
      except KeyError:
         pass

      # Change color names to numbers.
      for indx, item in enumerate(_colors.keys()):
         colors[indx] = colors[item]
         del colors[item]

      # Scatter data on the graph and display it.
      for indx, x in enumerate(X):
         try:
            if y is not None: # If labels are provided, then plot with labels as well.
               plt.scatter(x[0], x[1], s = 150, color = colors[y[indx]])
            else: # Otherwise, just plot all of the data points.
               plt.scatter(x[0], x[1], s = 150, color = 'crimson')
         except Exception as e:
            raise e

      # Display the plot.
      plt.show()

      # Return self (for stacked method calls).
      return self

   def plot(self, save = False):
      """Creates a scatter plot of provided data points and centroids, with class labels."""
      if self.data is None:
         # If the class has not been fit to data yet.
         raise ValueError("You need to fit the K-Means algorithm to training data before plotting it. If you want "
                          "to make a scatter plot of points, use KMeans.plot_scatter_data().")

      # Generate list of colors.
      _colors = mcolors.BASE_COLORS
      _colors.update(mcolors.TABLEAU_COLORS)
      _colors.update(mcolors.CSS4_COLORS)
      colors = _colors.copy()

      # Remove white entirely, that is reserved for the centroids.
      try:
         del _colors['w'], colors['w']
      except KeyError:
         pass

      # Change color names to numbers.
      for indx, item in enumerate(_colors.keys()):
         colors[indx] = colors[item]
         del colors[item]

      # Plot the data points with their colors.
      for classification in self.classes:
         # Determine class color.
         current_color = colors[classification]

         # Plot the individual point.
         for feature in self.classes[classification]:
            plt.scatter(feature[0], feature[1], color = current_color, s = 30)

      # Plot the centroids.
      for centroid in self.centroids:
         plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1], s = 130, marker = "X",
                     linewidths = 5, facecolor = 'white', edgecolors = 'black', linewidth = 1)

      # Display the plot.
      savefig = plt.gcf()
      plt.show()

      # If requested to save, then save.
      if save:
         try:
            savefig.savefig(save)
         finally:
            del savefig

      # Return self (for stacked method calls).
      return self


