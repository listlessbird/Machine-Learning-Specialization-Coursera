import copy
import math

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

print(f"Shape of features: {X_train.shape} type: {type(X_train)}")
print(X_train)
print(f"Shape of features: {y_train.shape} type: {type(y_train)}")
print(y_train)

b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")


def predict(x, w, b):
    return np.dot(x, w) + b


x_vec = X_train[0, :]

f_wb = predict(x_vec, w_init, b_init)
print(f"f_wb shape: {f_wb.shape} prediction: {f_wb}")


def compute_cost(x, y, w, b):
    cost = 0.0
    m = x.shape[0]

    for i in range(0, m):
        f_wb_i = np.dot(x[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2

    cost = cost / (2 * m)
    return cost


cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')


def compute_gradient(X, y, w, b):
    """
    compute gradient for linear regression
    """
    m, n = X.shape
    # n weights for n features
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(0, m):
        err = (np.dot(w, X[i]) + b) - y[i]

        for j in range(0, n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]

        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')


def gradient_descent(X, y, w_in, b_in, gradient_func, cost_func, alpha, num_iters):
    w = copy.deepcopy(w_in)
    b = b_in

    j_history = []

    for i in range(0, num_iters):
        dj_dw, dj_db = gradient_func(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            j_history.append(cost_func(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration : {i:4d}: Cost {j_history[-1]:8.2f}")

    return w, b, j_history


initial_w = np.zeros_like(w_init)
initial_b = 0.
iterations = 1000
alpha = 5.0e-7
w_final, b_final, J_hist = gradient_descent(X=X_train, gradient_func=compute_gradient, cost_func=compute_cost,y=y_train, w_in=initial_w, b_in=initial_b, alpha=alpha, num_iters=iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

m, _ = X_train.shape

for i in range(0, m):
    print(f"Prediction: {predict(X_train[i, :],w_final, b_final)}, target value: {y_train[i]}")
