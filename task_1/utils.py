import math

import numpy as np


def calc_coefficient(A, n, T, I):
    return A / n * pow(T, 1.5) * np.exp(-I / T)


def get_jacobi(alpha):
    jacobi_matrix = np.zeros((5, 5), np.float)
    jacobi_matrix[0][0] = -alpha[1] * alpha[4] / alpha[0] / alpha[0]
    jacobi_matrix[0][1] = alpha[4] / alpha[0]
    jacobi_matrix[0][2] = 0
    jacobi_matrix[0][3] = 0
    jacobi_matrix[0][4] = alpha[1] / alpha[0]
    jacobi_matrix[1][0] = 0
    jacobi_matrix[1][1] = -alpha[2] * alpha[4] / alpha[1] / alpha[1]
    jacobi_matrix[1][2] = alpha[4] / alpha[1]
    jacobi_matrix[1][3] = 0
    jacobi_matrix[1][4] = alpha[2] / alpha[1]
    jacobi_matrix[2][0] = 0
    jacobi_matrix[2][1] = 0
    jacobi_matrix[2][2] = -alpha[3] * alpha[4] / alpha[2] / alpha[2]
    jacobi_matrix[2][3] = alpha[4] / alpha[2]
    jacobi_matrix[2][4] = alpha[3] / alpha[2]
    jacobi_matrix[3][0] = 1
    jacobi_matrix[3][1] = 1
    jacobi_matrix[3][2] = 1
    jacobi_matrix[3][3] = 1
    jacobi_matrix[3][4] = 0
    jacobi_matrix[4][0] = 0
    jacobi_matrix[4][1] = 1
    jacobi_matrix[4][2] = 2
    jacobi_matrix[4][3] = 3
    jacobi_matrix[4][4] = -1

    return jacobi_matrix


def calc_rhs(A, n, T, alpha, I_1, I_2, I_3):
    rhs = np.zeros(5)
    rhs[0] = alpha[1] * alpha[4] / alpha[0] - calc_coefficient(A, n, T, I_1)
    rhs[1] = alpha[2] * alpha[4] / alpha[1] - calc_coefficient(A, n, T, I_2)
    rhs[2] = alpha[3] * alpha[4] / alpha[2] - calc_coefficient(A, n, T, I_3)
    rhs[3] = alpha[0] + alpha[1] + alpha[2] + alpha[3] - 1
    rhs[4] = alpha[1] + 2 * alpha[2] + 3 * alpha[3] - alpha[4]

    return rhs


def newton_method(iter_max, A, n, T, I_1, I_2, I_3, epsilon):

    num_iter = 0
    alpha = get_init()
    current_error = np.inf

    while current_error > epsilon:
        d_alpha = np.linalg.solve(
            get_jacobi(alpha),
            -calc_rhs(A, n, T, alpha, I_1, I_2, I_3)
        )
        alpha += d_alpha
        num_iter += 1
        current_error = np.linalg.norm(d_alpha)
        if num_iter > iter_max:
            break

    return alpha


def get_init():
    alpha = np.full(5, 0.1, np.float)
    k = 1e-1
    alpha[1] = k
    alpha[2] = pow(k, 2)
    alpha[3] = pow(k, 3)
    alpha[0] = 1 - alpha[1] - alpha[2] - alpha[3]
    alpha[4] = alpha[1] + 2 * alpha[2] + 3 * alpha[3]
    return alpha
