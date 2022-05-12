import sys

import pickle

import numpy as np
from rk.runge_kutta import RungeKutta
import matplotlib.pyplot as plt
# from model.model import PhotoionizationModel
from model.model2 import PhotoionizationModel
from model.consts import *

n = 425.4
r_array = np.linspace(R_sun_non_dim, r_non_dim, 10)
# r_array = np.linspace(r_non_dim, r_non_dim, 1)
model = PhotoionizationModel(n, r_array=r_array)

path = "/home/ilya/obwesos/sem_2/output.pickle"
with open(path, "rb") as f:
    data = pickle.load(f)

solution = data[0]
j_array = data[1]
current_time = data[2]
step = data[3]
iteration_number = data[4]
#
# a = 1

alpha_init = solution[-1, :, 0]
T_e_init = solution[-1, :, 1]
T_init = solution[-1, :, 2]

# alpha = solution[:, :, 0]
# T = solution[:, :, 2]
# T_e = solution[:, :, 1]

# t_linspace = np.linspace(0, iteration_number, iteration_number + 1)
# t_linspace_sparced = []

# alpha_init = 1e-5
# T_init = 1e-3
# T_e_init = 1e-3

model.initialize(alpha_init=alpha_init, T_e_init=T_e_init, T_init=T_init, is_array=True)


t_max = 1e9

init_step = step
t0 = current_time
tol = 1e-5
model.solve(t0=t0, time_max=t_max, init_step=init_step, tolerance=tol)
