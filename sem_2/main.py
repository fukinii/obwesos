import numpy as np
from rk.runge_kutta import RungeKutta
import matplotlib.pyplot as plt
from model.model import PhotoionizationModel
from model.consts import *

n = 425.4
r_array = np.linspace(R_sun_non_dim, r_non_dim, 100)
model = PhotoionizationModel(n, r_array=r_array)

alpha_init = 1e-5
T_init = 1e-3
T_e_init = 1e-3

model.initialize(alpha_init=alpha_init, T_e_init=T_e_init, T_init=T_init)

model.solve(t0=0., time_max=1e9, init_step=1e2, tolerance=1e-12)
