import sys
import pickle

import math

import numpy as np
from .consts import *
from integral.integral import integrate_from_points
from rk.runge_kutta import RungeKutta
from multiprocessing import Pool
from functools import partial

class PhotoionizationModel:

    def __init__(self, n, r_array):
        self.equationNumber = 3
        self.n = n
        self.r_array = r_array
        self.alpha_array = np.zeros((1, len(self.r_array)))
        self.T_array = np.zeros((1, len(self.r_array)))
        self.T_e_array = np.zeros((1, len(self.r_array)))
        self.j_array = np.zeros((1, len(self.r_array)))

        self.solution = np.zeros((1, len(self.r_array), 3))

        self.rk = RungeKutta()

    def initialize(self, alpha_init, T_e_init, T_init, is_array=False):
        if not is_array:
            self.alpha_array[0, :] = alpha_init
            self.T_array[0, :] = T_init
            self.T_e_array[0, :] = T_e_init

            self.solution[0, :, 0] = alpha_init
            self.solution[0, :, 1] = T_e_init
            self.solution[0, :, 2] = T_init

        else:
            self.alpha_array[0] = alpha_init
            self.T_array[0] = T_init
            self.T_e_array[0] = T_e_init

            self.solution[0, :, 0] = alpha_init
            self.solution[0, :, 1] = T_e_init
            self.solution[0, :, 2] = T_init



        a = 1

    def solve(self, t0, time_max, init_step, tolerance=1e-8, dump_file=False):

        current_time = t0
        step = init_step
        prev_t = current_time
        t_array = np.array([current_time])

        res = np.zeros((1, 3, len(self.r_array)))

        increased_tol_1_times = False
        increased_tol_2_times = False
        iteration_number = 0
        # pool = Pool(12)
        while current_time < time_max:
            prev_step = step
            t_array = np.append(t_array, prev_t + prev_step)
            prev_t = prev_t + prev_step

            next_res = np.zeros((3, len(self.r_array)))
            next_j = np.zeros((1, len(self.r_array)))
            next_step_min = 1e20

            # # with Pool(processes=12) as pool:
            # func = partial(self.parall_func, current_time, step, tolerance)
            # # next_ste_array, next_step_data = pool.map(func, np.arange(len(self.r_array)))
            # data = pool.map(func, np.arange(len(self.r_array)))
            #
            # # for data_i in data:
            # for r_index, r in enumerate(self.r_array):
            #     next_step = data[r_index][0]
            #     if next_step < next_step_min:
            #         next_step_min = next_step
            #
            #     tau_0 = self.calc_tau_0(r_index)
            #     next_j[0, r_index] = self.calc_j_v(tau_0, r_index)
            #     next_res[:, r_index] = data[r_index][1]
            #
            # # next_step_min = np.min(data[0])
            #
            #
            #
            # a = 1

            for r_index, r in enumerate(self.r_array):
                tau_0 = self.calc_tau_0(r_index)
                alpha = self.solution[-1, r_index, 0]
                T_e = self.solution[-1, r_index, 1]
                T = self.solution[-1, r_index, 2]

                def function(t, data):
                    return self.calc_rhs(alpha=data[0], T_e=data[1], T=data[2], tau_0=tau_0, r_index=r_index)

                next_step, next_data = self.rk.calc_single_adaptive_step(init_data=np.array([alpha, T_e, T]),
                                                                        t0=current_time, function=function,
                                                                        step=step,
                                                                        tolerance=tolerance)
                if next_step < next_step_min:
                    next_step_min = next_step
            # if r == 0.05:
            # if next_data[0] - 1 > 0:
            # if next_data[0] - 1 > 1e-5:
            #     print("Альфа слишком большой: ", next_data[0])
            #     assert False
            # else:
            # next_data[0] = 1.
            # if r_index == 0:


                    # "\nT = ", next_data[2])
                next_j[0, r_index] = self.calc_j_v(tau_0, r_index)
                next_res[:, r_index] = next_data
                # print("===============\nr = ", r,
                #     "\nalpha = ", next_data[0],
                #     "\nT_e = ", next_data[1],
                #     "\nj = ", next_j[0][r_index])
            ###################################################
            step = next_step_min
            next_res = next_res.T
            next_res_1 = np.reshape(next_res, (1, next_res.shape[0], next_res.shape[1]))
            self.j_array = np.concatenate((self.j_array, next_j), axis=0)

            self.solution = np.concatenate((self.solution, next_res_1), axis=0)

            # print(self.j_array)
            a = 1
            # next_alpha = next_res[0, :]
            # next_alpha_reshape = np.reshape(next_alpha, (1, len(next_alpha)))
            # self.alpha_array = np.concatenate((self.alpha_array, next_alpha_reshape),
            #                                   axis=0)
            # self.T_array = np.concatenate((self.T_array, next_res[1, :]), axis=0)
            # self.T_e_array = np.concatenate((self.T_e_array, next_res[2, :]), axis=0)

            # res = np.concatenate((res, next_res), axis=0)

            if iteration_number % 100 == 0:
                print("++++++++++++++++++++++++++++++++++++++\n")
                print("step = ", step,
                    "\nt = ", current_time, f'\n iter: {iteration_number}')
                print("\n++++++++++++++++++++++++++++++++++++++\n")

            # print(step)

            current_time = current_time + step

            # if current_time > 100 and not increased_tol_1_times:
            #     print("\n!!!!!!!!!!!!!!!Уменьшил точность!!!!!!!!!!!!!!!!!!!!!\т")
            #     tolerance = tolerance * 1e3
            #     # tolerance = tolerance * 2.3e4
            #     increased_tol_1_times = True

            # if current_time > 1e6 and not increased_tol_2_times:
            #     tolerance = tolerance * 2
            #     increased_tol_2_times = True

            iteration_number += 1
            if dump_file:
                if iteration_number % 1000 == 0:
                    data_to_dump = [self.solution, t_array, self.j_array, current_time, step, iteration_number]
                    with open('output__.pickle', 'wb') as f:
                        pickle.dump(data_to_dump, f)
                    print(f'iter: {iteration_number}, cur_time: {current_time} pickle saved')
                    del data_to_dump

        return self.solution

    def calc_rhs(self, alpha, T_e, T, tau_0, r_index):
        # print(T_e)
        if alpha > 1:
            assert False, "alpha > 1"
        if alpha < 0:
            assert False, "alpha < 0"
            # T = np.abs(T)
        if T < 0:
            assert False, "T < 0"
            # T = np.abs(T)
        if T_e < 0:
            assert False, "T_e < 0"
            # T_e = np.abs(T_e)
        res = np.zeros(self.equationNumber)
        j_ei = self.calc_j_ei(T_e=T_e)
        j_nu_ei = self.calc_j_nu_ei(T_e=T_e)
        K = self.calc_K(T_e=T_e)
        j_v = self.calc_j_v(tau_0, r_index)
        nu_e_mean = self.calc_v_e_mean(T_e=T_e)
        nu_e = self.calc_nu_e(v_e_mean=nu_e_mean, alpha=alpha, T_e=T_e)
        res[0] = ((1 - alpha) * j_v - alpha * alpha * self.n * j_nu_ei) + \
                 self.n * alpha * j_ei * ((1 - alpha) * K - self.n * alpha * alpha)
        res[1] = (T - T_e) * nu_e - (
                2. * I / 3 + T_e) * self.n * j_ei * ((1 - alpha) * K - self.n * alpha * alpha) + T_e * (
                         1 - 2. * self.calc_F(T_e=T_e) / 3) * alpha * self.n * j_nu_ei + (2. * T_v / 3. - T_e) * (
                         1 - alpha) / alpha * j_v
        res[2] = - (T - T_e) * alpha * nu_e

        # if res[0] < 0:
        #     assert False, "alpha < 0"
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
        # a = pow(T_e, 3. / 2.)
        # b = np.exp(-I / T_e)
        res = 3e21 * pow(T_e, 3. / 2.) * np.exp(-I / T_e)
        return res

    @staticmethod
    def calc_F(T_e):
        res = 0.64 + 0.11 * np.log10(I / T_e)
        # res = 0.64 + 0.11 * np.log(I / T_e)
        return res

    @staticmethod
    def calc_J(tau_0):
        res_ = -39.6 * (1 + 9.56e-7 * tau_0 ** 2) + 26.2 / (1 + 0.016 * tau_0)
        res = pow(10, res_)
        # res = np.exp(res_)
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

    def calc_tau_0(self, r_index):
        alpha = self.solution[-1, :r_index + 1, 0]
        # tau_0 = integrate_from_points(1 - self.alpha_array[-1, :r_index + 1], self.r_array[:r_index + 1])
        tau_0 = integrate_from_points(1 - alpha, self.r_array[:r_index + 1])
        tau_00 = R_0 * self.n * sigma_0_v

        return tau_0 * tau_00

    def calc_j_v(self, tau_0, r_index):
        j_nu_0 = self.calc_j_nu_0()
        J = self.calc_J(tau_0=tau_0)
        r = self.r_array[r_index]
        res = j_nu_0 * J / r / r
        # res = self.calc_j_nu_0() * self.calc_J(tau_0=tau_0) / self.r_array[-1] / self.r_array[-1]
        return res

    def parall_func(self, current_time, step, tolerance, r_index):
        tau_0 = self.calc_tau_0(r_index)
        alpha = self.solution[-1, r_index, 0]
        T = self.solution[-1, r_index, 2]
        T_e = self.solution[-1, r_index, 1]

        def function(t, data):
            return self.calc_rhs(alpha=data[0], T_e=data[1], T=data[2], tau_0=tau_0, r_index=r_index)

        next_step, next_data = self.rk.calc_single_adaptive_step(init_data=np.array([alpha, T_e, T]),
                                                                 t0=current_time, function=function,
                                                                 step=step,
                                                                 tolerance=tolerance)

        return next_step, next_data
