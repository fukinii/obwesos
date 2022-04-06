import numpy as np
from rk.runge_kutta import RungeKutta
import matplotlib.pyplot as plt

rk = RungeKutta()

t0 = 0
step = 0.1
n_steps = 1000

# init_data = 0

init_data = np.array([1., 0.])


def f(t, x):
    res = np.zeros_like(x)
    res[0] = x[1]
    res[1] = -x[0]
    return res


# res = rk.calc_fix_step(init_data, f, t0, step, n_steps)

res_ad, t_array = rk.calc_adaptive_step(init_data=init_data, function=f, t0=t0, t_last=100, init_step=0.1,
                                        tolerance=1e-9)
# arg_linspace= np.linspace(0, res_ad[len(res_ad) - 1], len(res_ad))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(t_array, res_ad[:, 1], label="res[0]")
plt.show()
