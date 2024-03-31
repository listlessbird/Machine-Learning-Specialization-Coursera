import copy
import math

import numpy as np
from utils import *
import matplotlib.pyplot as plt

x_train, y_train = load_data()


def compute_cost(x, y, w, b):
    # J(w,b) = 1/2m * (f_wb(x) - y)^2
    cost = 0.0
    m = x.shape[0]

    for i in range(0, m):
        f_wb = np.dot(w, x) + b
        cost += (f_wb[i] - y[i]) ** 2

    cost = cost / (2 * m)
    return cost


# initial_w = 2
# initial_b = 1
#
# cost = compute_cost(x_train, y_train, initial_w, initial_b)
# print(type(cost))
# print(cost)
# print(f"Cost at initial w: {cost:.3f}")


def compute_gradient(x, y, w, b):
    dj_dw = 0
    dj_db = 0

    m = x.shape[0]

    f_wb = np.dot(w, x) + b
    # single var regression
    for i in range(0, m):
        dj_db_i = f_wb[i] - y[i]
        dj_dw_i = dj_db_i * x[i]

        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


# initial_w = 0
# initial_b = 0
#
# tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
# print("Gradient at initial w, b (zeros):", tmp_dj_dw, tmp_dj_db)


def gradient_descent(
    x, y, w_in, b_in, cost_fn, grad_fn, learning_rate, iteration_count
):
    j_hist = []
    w_hist = []

    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(0, iteration_count):
        # get the gradients
        dj_dw, dj_db = grad_fn(x, y, w, b)

        # update params
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        # now calculate new cost with the new params

        if i < 10000:
            cost = cost_fn(x, y, w, b)
            j_hist.append(cost)

        # log every 10 times intervals

        if i % math.ceil(iteration_count / 10) == 0:
            w_hist.append(w)
            print(f"Iteration {i:4}: Cost {float(j_hist[-1]):8.2f} ")

    return w, b, j_hist, w_hist


initial_w = 0.0
initial_b = 0.0

iterations = 1500
alpha = 0.01

w, b, _, _ = gradient_descent(
    x_train,
    y_train,
    initial_w,
    initial_b,
    compute_cost,
    compute_gradient,
    alpha,
    iterations,
)
print("w,b found by gradient descent:", w, b)


predicted = np.dot(w, x_train) + b
plt.plot(x_train, predicted, c="b")

plt.scatter(x_train, y_train, marker="x", c="r")

predict1 = 3.5 * w + b
print("For population = 35,000, we predict a profit of $%.2f" % (predict1 * 10000))

predict2 = 7.0 * w + b
print("For population = 70,000, we predict a profit of $%.2f" % (predict2 * 10000))
plt.show()
