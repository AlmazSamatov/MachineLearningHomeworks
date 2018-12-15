import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_data(b1, b0, size, x_range=(-10, 10), noise_mean=0,
                  noise_std=1):
    """
          input:
          b1, b0 - true parameters of data
          size - size of data, numbers of samples
    x_range - tuple of (min, max) x-values
    noise_mean - noise mean value
    noise_std - noise standard deviation

          output:
          data_x, data_y - data features
          """
    noise = np.random.normal(noise_mean, noise_std, size)
    rnd_vals = np.random.rand(size)
    data_x = x_range[1] * rnd_vals + x_range[0] * (1 - rnd_vals)
    data_y = b1 * data_x + b0 + noise

    return data_x, data_y


def gradient_descent(x, y, lr, n_iterations):
    """
    Function for linear regression based on gradient descent
    Arguments:
        x - input data
        y - labels
        lr - learning rate
        n_iterations - number of iterations
    Returns:
        beta - array of predictors coefficients
    """
    N = float(len(x))
    b_0 = float(0)
    b_1 = float(0)
    for i in range(n_iterations):
        b_0_der = 0
        b_1_der = 0
        for j in range(len(x)):
            b_0_der += -(2 / N) * (y.values[j] - (b_0 + b_1 * x.values[j]))
            b_1_der += -(2 / N) * x.values[j] * (y.values[j] - (b_0 + b_1 * x.values[j]))
        b_0 = b_0 - lr * b_0_der
        b_1 = b_1 - lr * b_1_der
    return [b_0, b_1]


def animate(data_x, data_y, true_b1, true_b0, b1, b0, x_range=(-10, 10),
            label="Least squares"):
    plt.scatter(data_x, data_y)
    plt.plot([x_range[0], x_range[1]],
             [x_range[0] * true_b1 + true_b0, x_range[1] * true_b1 + true_b0],
             c="r", linewidth=2, label="True")
    plt.plot([x_range[0], x_range[1]],
             [x_range[0] * b1 + b0, x_range[1] * b1 + b0],
             c="g", linewidth=2, label=label)
    plt.legend()
    plt.show()


### Parameters for data generation ###
true_b1 = 2.5
true_b0 = -7
size = 100
x_range = (0, 10)
noise_mean = 0
noise_std = 1

# Generate the data
data_x, data_y = generate_data(true_b1, true_b0, size,
                               x_range=x_range,
                               noise_mean=noise_mean,
                               noise_std=noise_std)

# Predict data's parameters
b1, b0 = gradient_descent(data_x, data_y, 0.01, 1000)

# Visualize the data
print("true b1 : {}\ntrue b0 : {}".format(true_b1, true_b0))
print("calculated b1 : {}\ncalculated b0 : {}".format(b1, b0))
animate(data_x, data_y, true_b1, true_b0, b1, b0, x_range=x_range)
