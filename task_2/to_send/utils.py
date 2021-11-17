import numpy as np


def rungekutta4(f, x0, t):
    num_of_nodes = np.shape(t)[0]
    x = np.zeros((num_of_nodes, np.shape(x0)[0]))
    x[0] = x0
    print("num_of_nodes = ", num_of_nodes)
    print("h = ", t[1] - t[0])
    for i in range(num_of_nodes - 1):

        h = t[i + 1] - t[i]
        if i % 100000 == 0:
            print("i, x[i]: ", i, x[i])
        k1 = f(x[i], t[i])
        k2 = f(x[i] + k1 * h / 2., t[i] + h / 2.)
        k3 = f(x[i] + k2 * h / 2., t[i] + h / 2.)
        k4 = f(x[i] + k3 * h, t[i] + h)
        x[i + 1] = x[i] + (h / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x


def calc_tau(f, alpha):
    if np.max(np.abs(f)) == 0:
        return 1e-16
    else:
        return 1e-9 * np.linalg.norm(alpha) / np.max(np.abs(f))


def rungekutta4_t(f, x0, t_max):
    x = np.zeros((1, np.shape(x0)[0]))
    x[0] = x0
    t = 0
    while t < t_max:
        x_curr = x[-1]

        k1 = f(x_curr, t)
        h = calc_tau(k1, x_curr)
        print(t, h)
        k2 = f(x_curr + k1 * h / 2., t + h / 2.)
        k3 = f(x_curr + k2 * h / 2., t + h / 2.)
        k4 = f(x_curr + k3 * h, t + h)
        x_next = x_curr + (h / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

        x_next = x_next.reshape((1, 4))
        x = np.concatenate((x, x_next), axis=0)
        t = t + h

    return x


def calc_n_t(t, R_0=100., u=1e6, n_0=1e17):
    n = n_0 * (R_0 / (R_0 + u * t)) ** 3
    return n


def calc_T_t(t, R_0=100., u=1e6, T_0=2.):
    T = T_0 * (R_0 / (R_0 + u * t)) ** 2
    return T


def calc_k_1(t, A=6.06e21, I_1=5.98):
    T = calc_T_t(t)
    n = calc_n_t(t)
    k1 = A / n * T ** (3. / 2) * np.exp(-I_1 / T)
    return k1


def calc_k_2(t, A=6.06e21, I_2=18.83):
    T = calc_T_t(t)
    n = calc_n_t(t)
    k2 = A / n * T ** (3. / 2) * np.exp(-I_2 / T)
    return k2


def calc_k_3(t, A=6.06e21, I_3=28.4):
    T = calc_T_t(t)
    n = calc_n_t(t)
    k3 = A / n * T ** (3. / 2) * np.exp(-I_3 / T)
    return k3


def calc_j_1(t):
    T = calc_T_t(t)
    n = calc_n_t(t)
    return 8.75e-27 * n ** 2 / (T ** (9. / 2))


def calc_j_2(t):
    T = calc_T_t(t)
    n = calc_n_t(t)
    return 7.0e-26 * n ** 2 / (T ** (9. / 2))


def calc_j_3(t):
    T = calc_T_t(t)
    n = calc_n_t(t)
    return 2.36e-25 * n ** 2 / (T ** (9. / 2))


def calc_func(alpha_vector, t):
    func = np.zeros_like(alpha_vector)

    alpha_0 = alpha_vector[0]
    alpha_1 = alpha_vector[1]
    alpha_2 = alpha_vector[2]
    alpha_3 = alpha_vector[3]

    alpha = alpha_1 + 2 * alpha_2 + 3 * alpha_3

    j_1 = calc_j_1(t)
    j_2 = calc_j_2(t)
    j_3 = calc_j_3(t)

    k_1 = calc_k_1(t)
    k_2 = calc_k_2(t)
    k_3 = calc_k_3(t)

    func[0] = - alpha * j_1 * (alpha_0 * k_1 - alpha * alpha_1)
    func[1] = alpha * j_1 * (alpha_0 * k_1 - alpha * alpha_1) - alpha * j_2 * (alpha_1 * k_2 - alpha * alpha_2)
    func[2] = alpha * j_2 * (alpha_1 * k_2 - alpha * alpha_2) - alpha * j_3 * (alpha_2 * k_3 - alpha * alpha_3)
    func[3] = alpha * j_3 * (alpha_2 * k_3 - alpha * alpha_3)

    assert func[0] == func[0]
    return func
