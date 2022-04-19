import numpy as np


def integrate(func, x_linspace):
    sum = 0

    for i in range(len(x_linspace) - 1):
        x_i = x_linspace[i]
        x_i_plus_1 = x_linspace[i + 1]

        f_i = func(x_i)
        f_i_plus_1 = func(x_linspace[i + 1])

        dx = x_i_plus_1 - x_i
        sum += (f_i_plus_1 + f_i) / 2 * dx

    return sum


def integrate_from_points(y_linspace, x_linspace):
    sum = 0

    for i in range(len(x_linspace) - 1):
        x_i = x_linspace[i]
        x_i_plus_1 = x_linspace[i + 1]

        y_i = y_linspace[i]
        y_i_plus_1 = y_linspace[i + 1]

        dx = x_i_plus_1 - x_i
        sum += (y_i_plus_1 + y_i) / 2 * dx

    return sum
