import math

import numpy as np
from .butcher_table import ButcherTable4


# import butcher_table as bt


class RungeKutta:
    def __init__(self):
        self.butcherTable = ButcherTable4()
        self.max_iters = 100

    def calc_fix_step(self, init_data, function, t0, step, n_steps):
        res = np.repeat(np.reshape(init_data, (1, init_data.shape[0])), n_steps + 1, axis=0)
        t_array = np.zeros(n_steps + 1)
        t_array[0] = t0
        for i in range(n_steps):
            t = t0 + step * i
            res[i + 1] = self.calc_single_fix_step(init_data=res[i], t0=t, function=function, step=step)
            t_array[i] = t
        t_array[len(t_array) - 1] = t0 + step * n_steps
        return res, t_array

    def calc_adaptive_step(self, init_data, function, t0, t_last, init_step, tolerance):

        res = np.reshape(init_data, (1, init_data.shape[0]))
        t = t0
        step = init_step
        prev_step = init_step
        prev_t = t
        t_array = np.array([t])
        while t < t_last:
            prev_step = step
            t_array = np.append(t_array, prev_t + prev_step)
            prev_t = prev_t + prev_step
            step, res_i = self.calc_single_adaptive_step(init_data=res[-1], t0=t, function=function,
                                                         step=step, tolerance=tolerance)
            res_i = np.reshape(res_i, (1, res_i.shape[0]))
            res = np.append(res, res_i, axis=0)

            t = t + step
        # t_array = np.append(t_array, t)
        return res, t_array

    def calc_single_adaptive_step(self, init_data, t0, function, step, tolerance):

        next_data_one_step = init_data
        iter = 0
        # while iter < self.max_iters:
        data_half_step = np.copy(next_data_one_step)
        next_data_one_step = self.calc_single_fix_step(init_data=next_data_one_step, t0=t0, function=function,
                                                       step=step)

        data_half_step = self.calc_single_fix_step(init_data=data_half_step, t0=t0, function=function,
                                                   step=step / 2)
        data_half_step = self.calc_single_fix_step(init_data=data_half_step, t0=t0, function=function,
                                                   step=step / 2)

        # factor = 1. / (pow(2, self.butcherTable.order) - 1)
        factor = 1. / (1 - pow(2, - self.butcherTable.order))
        errorVector = (data_half_step - next_data_one_step) * factor
        # error = errorVector.norm() / next_data_one_step.norm()
        error = np.linalg.norm(errorVector) / np.linalg.norm(next_data_one_step)

        # if error == 0.:
        # assert False, "error = 0."
        # T = np.abs(T)
        # a = 1

        # if math.isnan(error):
        # assert False, "error = nan."
        # a = 1

        local_factor = pow(tolerance / error, 1. / self.butcherTable.order)
        next_data_one_step += errorVector
        step = step * local_factor

        a = 1
        # if local_factor >= 1.:
        #     next_data_one_step += errorVector
        #     step = step * local_factor
        #     break
        # else:
        #     step = step * local_factor
        #     next_data_one_step = init_data
        # iter += 1

        # print("error = ", error, "\n step = ", step,
        #       "\nalpha = ", next_data_one_step[0])
        #       # "\nT_e = ", next_data_one_step[1],
        #       # "\nT = ", next_data_one_step[2])

        return step, next_data_one_step

    def calc_single_fix_step(self, init_data, t0, function, step):
        cols_num = self.butcherTable.column.shape[0]
        rows_num = self.butcherTable.row.shape[0]

        kArray = np.repeat(np.reshape(init_data, (1, init_data.shape[0])), cols_num, axis=0)
        for row_idx in range(rows_num):
            tmp_data = init_data
            tmp_time = t0 + step * self.butcherTable.column[row_idx]
            for column_idx in range(row_idx):
                tmp_data = tmp_data + kArray[column_idx] * self.butcherTable.matrix[row_idx, column_idx] * step

            if tmp_data[0] - 1 > 0.:
                tmp_data[0] = 1.
            # if tmp_data[0] < 0.:
            #     tmp_data[0] = 1e-10
            # if tmp_data[1] < 0.:
            #     tmp_data[1] = 1e-10
            # if tmp_data[2] < 0.:
            #     tmp_data[2] = 1e-10
            kArray[row_idx] = function(tmp_time, tmp_data)
        tmp_data = np.zeros_like(init_data)
        for idx, kArray_i in enumerate(kArray):
            tmp_data = tmp_data + kArray_i * self.butcherTable.row[idx]

        tmp_data *= step
        tmp_data += init_data
        if tmp_data[0] - 1 > 1e-5:
            tmp_data[0] = 1.
        #     print("Альфа слишком большой: ", tmp_data[0])
        #     assert False
        # else:
        #     tmp_data[0] = 1.
        return tmp_data
