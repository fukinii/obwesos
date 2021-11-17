import numpy as np


def calc_dichotomy(func, temperature, n, left, right, tolerance, max_num_of_iters=10000):
    num_of_iters = 0
    while right - left > tolerance:

        assert left <= right, "левая точка зашла за правую"

        mid = (left + right) / 2
        if np.sign(func(mid, temperature, n)) == np.sign(func(left, temperature, n)):
            left = mid
        else:
            right = mid

        num_of_iters += 1
        if num_of_iters > max_num_of_iters:
            break
    return (left + right) / 2


def calc_right_part(alpha, temperature, n, a_coef=6.06e21, i_1=5.98, i_2=18.83, i_3=28.4):
    alpha0 = -3 * (a_coef / n) ** 3 * temperature ** 4.5 * np.exp(-(i_1 + i_2 + i_3) / temperature)

    alpha1 = (a_coef / n) ** 3 * temperature ** 4.5 * np.exp(-(i_1 + i_2 + i_3) / temperature) - 2 * (
            a_coef / n) ** 2 * temperature ** 3 * np.exp(-(i_1 + i_2) / temperature)

    alpha2 = (a_coef / n) ** 2 * temperature ** 3 * np.exp(-(i_1 + i_2) / temperature) - (
            a_coef / n) * temperature ** 1.5 * np.exp(-i_1 / temperature)

    alpha3 = (a_coef / n) * temperature ** 1.5 * np.exp(-i_1 / temperature)
    alpha4 = 1

    return alpha4 * alpha ** 4 + alpha3 * alpha ** 3 + alpha2 * alpha ** 2 + alpha1 * alpha + alpha0


def calc_coefficient(n, temperature, i_coef, a_coef=6.06e21):
    return a_coef / n * pow(temperature, 1.5) * np.exp(-i_coef / temperature)


def solve_alpha(n, temperature, tolerance, a_coef=6.06e21, i_1=5.98, i_2=18.83, i_3=28.4):
    """
    Функция для решения системы на альфы. В данной реализации через альфу выражены все остальные компоненты,
    которые находятся после определения самой альфы методом дихотомии.

    :param i_3: константа, равная 28.4 эВ
    :param i_1: константа, равная 5.98 эВ
    :param i_2: константа, равная 18.83 эВ
    :param a_coef: константа, равная 6.06e21
    :param n: концентрация в 1/см^3
    :param temperature: температрура в Эв
    :param tolerance: точность метода деления пополам
    """

    # Считаем альфу
    alpha = calc_dichotomy(calc_right_part,
                           temperature=temperature,
                           n=n,
                           left=1e-7,
                           right=3,
                           tolerance=tolerance)

    # Определяем множители с экспонентой
    coef_in_i1 = calc_coefficient(n, temperature, i_1)
    coef_in_i2 = calc_coefficient(n, temperature, i_2)
    coef_in_i3 = calc_coefficient(n, temperature, i_3)

    # Пересчитываем другие альфы через найденную альфу
    alpha0 = 1 / (
            1 + coef_in_i1 / alpha + coef_in_i1 * coef_in_i2 / alpha ** 2 +
            coef_in_i1 * coef_in_i2 * coef_in_i3 / alpha ** 3)
    alpha1 = alpha0 / alpha * coef_in_i1
    alpha2 = alpha1 / alpha * coef_in_i2
    alpha3 = alpha2 / alpha * coef_in_i3

    return np.array([alpha0, alpha1, alpha2, alpha3, alpha])
