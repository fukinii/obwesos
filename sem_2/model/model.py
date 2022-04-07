import numpy as np
from .consts import *


class PhotoionizationModel:

    def __init__(self):
        self.equationNumber = 3

        pass

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

    @staticmethod
    def calc_nu_e(n, v_e_mean, alpha, T_e):
        res = n * 2 * m_e / m * (
                4. / 3. * sigma_e_0 * v_e_mean * (1 - alpha) +
                4 * np.sqrt(2 * np.pi) / 3 * alpha * Lambda * pow(e, 4) / (np.sqrt(m_e) * pow(1.6e-12 * T_e, 3. / 2)))
        return res

    def calc_j_v(self):
        pass
