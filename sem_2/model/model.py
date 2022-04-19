import numpy as np
from .consts import *
from sem_2.integral.integral import integrate_from_points
from sem_2.rk.runge_kutta import RungeKutta


class PhotoionizationModel:

    def __init__(self, n, r_array):
        self.equationNumber = 3
        self.n = n
        self.r_array = r_array
        self.alpha_array = np.zeros((1, len(self.r_array)))
        self.T_array = np.zeros((1, len(self.r_array)))
        self.T_e_array = np.zeros((1, len(self.r_array)))

        self.rk = RungeKutta()

    def initialize(self, alpha_init, T_e_init, T_init):
        self.alpha_array[0, :] = alpha_init
        self.T_array[0, :] = T_init
        self.T_e_array[0, :] = T_e_init
        a = 1

    def solve(self, t0, time_max, init_step, tolerance=1e-8):

        current_time = t0
        step = init_step
        prev_t = current_time
        t_array = np.array([current_time])

        res = np.zeros((1, 3, len(self.r_array)))

        while current_time < time_max:
            tau_0 = self.calc_tau_0()
            prev_step = step
            t_array = np.append(t_array, prev_t + prev_step)
            prev_t = prev_t + prev_step

            next_res = np.zeros((3, len(self.r_array)))

            for r_index, r in enumerate(self.r_array):
                alpha = self.alpha_array[-1, r_index]
                T = self.T_array[-1, r_index]
                T_e = self.T_e_array[-1, r_index]

                def function(t, data):

                    return self.calc_rhs(alpha=data[0], T_e=data[1], T=data[2], tau_0=tau_0)

                step, next_data = self.rk.calc_single_adaptive_step(init_data=np.array([alpha, T_e, T]),
                                                                    t0=current_time, function=function, step=step,
                                                                    tolerance=tolerance)
                next_res[:, r_index] = next_data

            self.alpha_array = np.concatenate((self.alpha_array, next_res[0, :]), axis=0)
            self.T_array = np.concatenate((self.T_array, next_res[0, :]), axis=0)
            self.T_e_array = np.concatenate((self.T_e_array, next_res[0, :]), axis=0)

            res = np.concatenate((res, next_res), axis=0)
            print(current_time)
        return res

    def calc_rhs(self, alpha, T_e, T, tau_0):
        res = np.zeros(self.equationNumber)
        j_ei = self.calc_j_ei(T_e=T_e)
        j_nu_ei = self.calc_j_nu_ei(T_e=T_e)
        K = self.calc_K(T_e=T_e)
        j_v = self.calc_j_v(tau_0)
        nu_e_mean = self.calc_v_e_mean(T_e=T_e)
        nu_e = self.calc_nu_e(v_e_mean=nu_e_mean, alpha=alpha, T_e=T_e)
        res[0] = ((1 - alpha) * j_v - alpha * alpha * self.n * j_nu_ei) + \
                 self.n * alpha * j_ei * ((1 - alpha) * K - self.n * alpha * alpha)
        res[1] = (T - T_e) * nu_e - (
                2. * I / 3 + T_e) * self.n * j_ei * ((1 - alpha) * K - self.n * alpha * alpha) + T_e * (
                         1 - 2. * self.calc_F(T_e=T_e) / 3) * alpha * self.n * j_nu_ei + (2. * T_v / 3. - T_e) * (
                         1 - alpha) / alpha * j_v
        res[2] = - (T - T_e) * alpha * nu_e
        return res

    @staticmethod
    def calc_j_nu_0():
        res = 15 * Phi_sun * pow(betta, 3) * sigma_0_v / (4 * pow(np.pi, 5) * 1.6e-12 * T_v * L ** 2 * a ** 2)
        return res

    @staticmethod
    def calc_j_nu_ei(T_e):
        res = 2.7e-13 / pow(T_e, 3. / 4.)
        return res

    @staticmethod
    def calc_j_ei(T_e):
        res = 8.75e-27 / pow(T_e, 9. / 2.)
        return res

    @staticmethod
    def calc_K(T_e):
        res = 3e21 * pow(T_e, 3. / 2.) * np.exp(-I / T_e)
        return res

    @staticmethod
    def calc_F(T_e):
        res = 0.64 + 0.11 * np.log10(I / T_e)
        return res

    @staticmethod
    def calc_J(tau_0):
        res_ = -39.6 * (1 + 9.56e-7 * tau_0 ** 2) + 26.2 / (1 + 0.016 * tau_0)
        res = pow(10, res_)
        return res

    def calc_nu_e(self, v_e_mean, alpha, T_e):
        res = self.n * 2 * m_e / m * (
                4. / 3. * sigma_e_0 * v_e_mean * (1 - alpha) +
                4 * np.sqrt(2 * np.pi) / 3 * alpha * Lambda * pow(e, 4) / (np.sqrt(m_e) * pow(1.6e-12 * T_e, 3. / 2)))
        return res

    @staticmethod
    def calc_v_e_mean(T_e):
        res = np.sqrt(8 * (1.6e-12 * T_e) / (np.pi * m_e))
        return res

    def calc_tau_0(self):
        tau_0 = integrate_from_points(1 - self.alpha_array[-1, :], self.r_array)
        return tau_0

    def calc_j_v(self, tau_0):
        res = self.calc_j_nu_0() * self.calc_J(tau_0=tau_0) / r_non_dim / r_non_dim

        return res
