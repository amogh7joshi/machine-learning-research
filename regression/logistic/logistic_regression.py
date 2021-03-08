#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression(object):
   def __init__(self):
      # Epsilon factor to prevent division by zero.
      self._epsilon = 1e-7

      # Other general class values which need to be initialized.
      self._weights = None
      self.data = None
      self._is_data_intercepted = False

   def __len__(self):
      # Return the length of the weight array.
      if self._weights is None:
         raise NotImplementedError("You need to fit the logistic regression to training data "
                                   "to initialize the weights and therefore give them a length.")
      else:
         return self._weights.shape[0]

   def __str__(self):
      # Return the class weights.
      return " ".join(f"{item:.2f}" for item in self._weights)

   @property
   def epsilon(self):
      """Method to view class epsilon value."""
      return self._epsilon

   @epsilon.setter
   def epsilon(self, value):
      """Method to set class epsilon value (cannot be done on __init__)."""
      self._epsilon = value

   @property
   def weights(self):
      """Method to view class weights values."""
      return self._weights

   @weights.setter
   def weights(self, weights):
      """A method to set the class weights (if you already have them)."""
      self._weights = weights

   @staticmethod
   def _assert_sizes(*args):
      """Ensure that input arrays have the same shape/length."""
      default_length = len(args[0])
      for value in args:
         if len(value) != default_length:
            raise ValueError(f"The inputted arrays have different lengths, "
                             f"e.g. {len(value)} and {default_length}.")

   @staticmethod
   def _convert_to_numpy(*args):
      """Internal decorator, converts inputted arguments to a usable NumPy array format."""
      converted_args = []
      for item in args:
         if isinstance(item, np.ndarray):
            converted_args.append(item)
         elif isinstance(item, (int, float)):
            converted_args.append(np.atleast_1d(item))
         else:
            converted_args.append(np.array(item))
      return converted_args

   def sigmoid(self, z):
      """A convenience internal method to return sigmoid(X)."""
      return 1 / (1 + np.exp(self.epsilon - z))

   def binary_crossentropy(self, y_pred, y_true):
      """A convenience internal method to return the binary crossentropy
      loss between predicted/true values (during gradient descent)."""
      y_true, y_pred = self._convert_to_numpy(y_true, y_pred)
      self._assert_sizes(y_true, y_pred)

      # Clip to the epsilon value.
      y_pred = np.clip(y_pred, self.epsilon, 1. - self.epsilon)

      # Get the batch size and return the actual loss.
      batch_size = y_pred.shape[0]
      return -np.sum(y_true * np.log(y_pred + 1e-9)) / batch_size

   def generate_scatter_data(self, amount, seed = None):
      """Generates scatter data for the logistic regression algorithm (and plotting)."""
      # Set a specific seed to keep generating the same data.
      if seed is not None:
         try: # Set the actual seed.
            np.random.seed(seed)
         except Exception as e:
            # Except invalid types or values.
            raise e
      else:
         # There is no seed necessary
         np.random.seed(None)

      # Create multivariate normal distribution values.
      choices1 = np.random.randint(0, 5, size = (2, ))
      choices2 = np.random.randint(6, 10, size = (2, ))

      # Create the data for two different classes.
      class1 = np.random.multivariate_normal([choices1[0], choices1[1]], [[1, .75], [.75, 1]], amount)
      label1 = np.zeros(amount)
      class2 = np.random.multivariate_normal([choices2[0], choices2[1]], [[1, .75], [.75, 1]], amount)
      label2 = np.ones(amount)

      # Create stacked data and labels.
      features = np.vstack((class1, class2)).astype(np.float32)
      labels = np.hstack((label1, label2))

      # Set the data to the class.
      self.data = (features, labels)

      # Return the features and labels.
      return features, labels

   def plot_scatter_data(self, *args):
      """Creates a two-dimensional scatter plot of data points."""
      args = self._convert_to_numpy(*args)
      self._assert_sizes(args)

      # For binary logistic regression, there should only be two classes.
      if len(args) != 2:
         # In this case, we have received multiple arrays that are incorrect.
         raise ValueError(f"Expected only two arrays, one containing data and one containing "
                          f"labels, instead received {len(args)} arrays. ")

      # Validate the rest of the data features.
      if np.size(args[0], -1) != 2:
         raise ValueError(f"Expected a two-dimensional data array, instead got {np.size(args[0], -1)}.")
      if not np.array_equal(args[1], args[1].astype(np.bool)):
         raise ValueError("The label array should be binary (0s and 1s), instead got multiple classes.")

      # Plot the data.
      plt.figure(figsize = (12, 8))
      plt.scatter(args[0][:, 0], args[0][:, 1], c = args[1], alpha = 0.4)

      # Display the plot.
      plt.show()

      # Convenience return for stacked method calls.
      return self

   def _gather_data(self, X = None, y = None):
      """Gets the data from provided values and returns it."""
      # Cases to determine the data being used.
      if self.data is not None:
         # The class has data saved already (from generate_scatter_data, etc.)
         X, y = self.data[0], self.data[1]
      elif isinstance(X, tuple) and y is None:
         # There may be a single tuple containing both X and y.
         X, y = X[0], X[1]
      else:
         # Data is just inputted as normal.
         X, y = X, y

      # Return the X and y values.
      return X, y

   def _initialize_weights(self, X):
      """Initializes the class weights (or re-initializes them for new data)."""
      del self._weights
      self._weights = np.zeros(X.shape[1])

   def _update_weights(self, X, y_pred, y_true, lr, l2_coef):
      """Updates the class weights for logistic regression."""
      # Calculate the gradient between the expected and predictions.
      discrepancy = y_true - y_pred
      grad = np.dot(X.T, discrepancy)

      # Apply regularization if a coefficient is provided.
      if l2_coef is not None:
         grad = l2_coef * grad + np.sum(self._weights)

      # Update the class weights.
      self._weights += grad * lr

   def fit(self, X = None, y = None, epochs = 10000, lr = 0.001, verbose = True,
           add_intercept = True, l2_coef = 0.5, override_callback = False):
      """Fits the logistic regression algorithm to the provided data and labels."""
      # Dispatch to gather data.
      X, y = self._gather_data(X, y)

      # Add an intercept value (for smoothing data).
      if add_intercept:
         intercept = np.ones((X.shape[0], 1))
         X = np.hstack((intercept, X))
         # Tracker for evaluation method.
         self._is_data_intercepted = True

      # Dispatch to initialize weights.
      self._initialize_weights(X)

      # Create a loss tracker for early stopping.
      loss_tracker = []

      # Iterate over each epoch.
      for epoch in range(epochs):
         # Calculate the current prediction.
         predictions = self.sigmoid(np.dot(X, self._weights))

         # A default early stopping criterion, stops training if nothing improves
         # or if the loss actually starts to go up instead of down.
         if not override_callback:
            if len(loss_tracker) < (epochs // 10):
               loss_tracker.append(self.binary_crossentropy(predictions, y))
            else:
               loss_tracker = loss_tracker[1:]
               loss_tracker.append(self.binary_crossentropy(predictions, y))

               # Determine if the loss is not changing.
               if np.all(np.isclose(loss_tracker, loss_tracker[0])):
                  print(f"Stopping training early because loss is not decreasing. "
                        f"Final Loss: {self.binary_crossentropy(predictions, y)}")
                  break

               # Determine if the loss is actually going up.
               if not np.diff(np.array(loss_tracker)).all() > 0:
                  print(f"Stopping training early because loss is increasing. "
                        f"Final Loss: {self.binary_crossentropy(predictions, y)}")
                  break

         # Dispatch to weight updating method.
         self._update_weights(X, predictions, y, lr = lr, l2_coef = l2_coef)

         # Print out the binary-crossentropy loss if necessary.
         if verbose:
            if epoch % (epochs / 20) == 0:
               print(f"Epoch {epoch}\t Loss: {self.binary_crossentropy(predictions, y)}")

      # Convenience return for stacked method calls.
      return self

   def evaluate(self, X = None, y = None):
      """Evaluates the logistic regression algorithm on the data (accuracy and loss)."""
      # Cases to determine the data being used.
      X, y = self._gather_data(X, y)

      # Determine if there is an intercept necessary.
      if self._is_data_intercepted:
         intercept = np.ones((X.shape[0], 1))
         X = np.hstack((intercept, X))

      # Calculate the predictions.
      total_predictions = np.round(self.sigmoid(np.dot(X, self._weights)))

      # Calculate the accuracy and loss.
      accuracy = np.sum(total_predictions == y).astype(np.float) / len(total_predictions)
      loss = self.binary_crossentropy(total_predictions, y)

      # Print out the accuracy.
      print(f"Accuracy: {accuracy * 100:.2f}%\t Loss: {loss}")

      # Convenience return for stacked method calls.
      return self

   def predict(self, value):
      """Predicts the class of a piece of data with the trained algorithm."""
      if self.data is None:
         raise ValueError("You need to fit the algorithm to data before trying to predict.")
      return self.sigmoid(np.dot(value, self._weights))

   def plot_evaluation(self, X = None, y = None):
      """Plots the correct/incorrect predictions."""
      X, y = self._gather_data(X, y)

      # Get the predictions.
      if self._is_data_intercepted:
         # Determine if there is an intercept necessary.
         intercept = np.ones((X.shape[0], 1))
         X_intercepted = np.hstack((intercept, X))
         total_predictions = np.round(self.sigmoid(np.dot(X_intercepted, self._weights)))
      else:
         # Otherwise, just use the regular data.
         total_predictions = np.round(self.sigmoid(np.dot(X, self._weights)))

      # This time, we want to generate manual colors (green for correct, red for wrong).
      discrepancies = np.equal(y, total_predictions)
      color_array = []
      for item in discrepancies:
         if item:
            color_array.append('tab:green')
         else:
            color_array.append('tab:red')

      # Create the scatter plot.
      plt.figure(figsize = (12, 8))
      plt.scatter(X[:, 0], X[:, 1], c = color_array, alpha = 0.4)

      # Display the plot.
      plt.show()

      # Convenience return for stacked method calls.
      return self


