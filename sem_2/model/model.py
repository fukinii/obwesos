import numpy as np
from .consts import *


class PhotoionizationModel:

    def __init__(self, n):
        self.equationNumber = 3
        self.n = n

        pass

    def calc_rhs(self, alpha, T_e, T, r, t):
        res = np.zeros(self.equationNumber)
        j_ei = self.calc_j_ei(T_e=T_e)
        j_nu_ei = self.calc_j_nu_ei(T_e=T_e)
        K = self.calc_K(T_e=T_e)
        j_v = self.calc_j_v(r, t)
        nu_e_mean = self.calc_v_e_mean(T_e=T_e)
        nu_e = self.calc_nu_e(v_e_mean=nu_e_mean, alpha=alpha, T_e=T_e)
        res[0] = ((1 - alpha) + j_v - alpha * alpha * self.n * j_nu_ei) + \
                 self.n * alpha * j_ei * ((1 - alpha) * K - self.n * alpha(alpha))
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

    @staticmethod
    def calc_tau_0():
        pass

    @staticmethod
    def calc_j_v(r, t):
        return 1.
