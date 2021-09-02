#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression(object):
   """The linear regression algorithm, implemented from scratch."""
   def __init__(self):
      self.theta = np.random.random(2)

   def __str__(self):
      return f"{self.theta[0]:.5f} + {self.theta[1]:.5f} * X"

   @staticmethod
   def assert_sizes(*arrays):
      """Ensure that arrays are the same size."""
      default_len = len(arrays[0])
      for array in arrays:
         if len(array) != default_len:
            raise ValueError(f"Training data arrays should have the same sizes, got one with length "
                             f"{default_len} but got another with length {len(array)}.")

   def hypothesis(self, X):
      """Linear regression hypothesis: h(Ø) = Ø_o + Ø_1 * x"""
      return self.theta[0] + self.theta[1] * X

   @staticmethod
   def cost(y_true, y_pred):
      """Linear regression cost function: J(Ø) = sum(y_pred - y_true)^2 """
      return pow((y_pred - y_true), 2)

   def fit(self, X, y, lr = 0.01, epochs = 100, verbose = True, visualize = True):
      """Linear regression training method, fits function to training data."""
      self.assert_sizes(X, y)

      # Determine when to output cost values (20 times total).
      if verbose:
         consistency = int(epochs / 20)

      # Iterate over epochs.
      for epoch in range(epochs):
         # Set default epoch values.
         cost = 0
         dtheta0 = 0
         dtheta1 = 0

         # Iterate over training data.
         for indx in range(len(X)):
            # Determine predicted y value.
            y_pred = self.hypothesis(X[indx])

            # Find the cost value.
            cost += self.cost(y[indx], y_pred)

            # Find the two ∂/∂Ø values.
            dtheta0 += lr * (y_pred - y[indx])
            dtheta1 += lr * (y_pred - y[indx]) * X[indx]

         # Update linear regression parameter values.
         cost /= len(X)
         dtheta0 /= len(X)
         dtheta1 /= len(X)

         self.theta[0] = self.theta[0] - dtheta0
         self.theta[1] = self.theta[1] - dtheta1

         # Output epoch values.
         if verbose:
            if (epoch + 1) % consistency == 0:
               if epoch + 1 > 9999:
                  print(f"Epoch: {epoch + 1}\t Cost: {cost}")
               else:
                  print(f"Epoch: {epoch + 1}\t\t Cost: {cost}")

      # If you want to visualize, visualize the graph.
      if visualize:
         self.plot(X, y)

   def predict(self, X):
      """Linear regression prediction method, applies hypothesis to list of inputs."""
      # If a list of values is provided, return all of them.
      if isinstance(X, (list, tuple, np.ndarray)):
         output_list = []
         for value in X:
            output_list.append(self.hypothesis(value))
         return output_list
      else:
         return self.hypothesis(X)

   def plot(self, X = None, y = None):
      """Visualize the linear regression, optionally with provided data."""
      # Because of the potential ValueError with NumPy, this excessive syntax is necessary.
      if (X is None and y is not None) or (y is None and X is not None):
         raise ValueError(
            "You need to provide both X and y values if you want to plot with data.")
      if X is not None and y is not None: # Ensure X and y are the same size.
         self.assert_sizes(X, y)

      # Plot the data points if provided.
      if X is not None and y is not None:
         plt.scatter(X, y, c = "BLUE")

      # Create X and y values to plot for the linear regression graph.
      if X is not None and y is not None: # If data is provided, only show the graph in that range.
         x_values = np.linspace(min(X) - 1, max(X) + 1)
         y_values = self.hypothesis(x_values)
      else:
         x_values = np.linspace(0, 100)
         y_values = self.hypothesis(x_values)

      # Plot the linear regression graph.
      plt.plot(x_values, y_values)

      # Show the graph.
      plt.show()







