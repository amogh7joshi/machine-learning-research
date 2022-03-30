#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Polynomial Regression with the gradient descent algorithm.
# Trying to determine the issues with using the gradient descent algorithm for polynomial regression.
# NOTED ISSUES:
# The algorithm seems to only try and match the last value in the data, e.g. if you provide an outlier as the
# last piece of data, it will only try to match that piece of data. This is likely due to the fact that because
# certain terms in polynomial regression have greater exponents, those terms have higher prevalence during the
# actual gradient descent computations. So, gradient descent is likely not the best algorithm for this.

class PolynomialRegression(object):
    """The polynomial regression algorithm, implemented from scratch."""
    def __init__(self, rank = 2):
        if rank < 1:
            raise ValueError("You need to provide a rank of 1 or greater, or else nothing is being implemented.")
        self.rank = rank + 1
        self.theta = np.random.random(rank + 1)

    def __str__(self):
        return_str = f"{self.theta[0]:.5f} + {self.theta[1]:.5f} * X"
        for i in range(2, self.rank):
            return_str += f" + {self.theta[i]:.5f} * X ^ {i}"
        return return_str

    @staticmethod
    def assert_sizes(*arrays):
        """Ensure that arrays are the same size."""
        default_len = len(arrays[0])
        for array in arrays:
            if len(array) != default_len:
                raise ValueError(f"Training data arrays should have the same sizes, got one with length "
                                 f"{default_len} but got another with length {len(array)}.")

    def hypothesis(self, X):
        """Polynomial regression hypothesis: h(Ø) = Ø_o + Ø_1 * x + Ø_2 * x ^ 2, etc."""
        return (1 / (2 * self.rank)) * sum(self.theta[i] * (X ** i) for i in range(self.rank))

    @staticmethod
    def cost(y_true, y_pred):
        """Polynomial regression cost function: J(Ø) = sum(y_pred - y_true)^2 """
        return pow((y_pred - y_true), 2)

    def fit(self, X, y, lr = 0.01, epochs = 100, verbose = True, visualize = True):
        """Polynomial regression training method, fits function to training data."""
        self.assert_sizes(X, y)

        # Determine when to output cost values (20 times total).
        if verbose:
            consistency = int(epochs / 20)

        # Iterate over epochs.
        for epoch in range(epochs):
            # Set default epoch values.
            cost = 0
            dtheta = np.empty(shape = (self.rank, ), dtype = float)

            # Iterate over training data.
            for indx in range(len(X)):
                # Determine predicted y value.
                y_pred = self.hypothesis(X[indx])

                # Find the cost value.
                cost += self.cost(y[indx], y_pred)

                # Find the ∂/∂Ø values.
                for i in range(self.rank):
                    # Create the individual value over a range.
                    value = 0
                    for j in range(i + 1):
                        value += (y_pred - y[indx]) * (X[indx] ** j)

                    # Multiply by learning rate and add to the dtheta value.
                    dtheta[i] = lr * value

            if epoch % 10 == 0:
                print(dtheta)

            # Update polynomial regression parameter values.
            cost /= len(X)
            for i in range(len(dtheta)):
                dtheta[i] = dtheta[i] / len(X)

            for i in range(self.rank):
                self.theta[i] = self.theta[i] - dtheta[i]

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
        if (X is None and y is not None) or (y is None and X is not None): # Both X and y need to be provided, or neither.
            raise ValueError("You need to provide both X and y values if you want to plot with data.")
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
