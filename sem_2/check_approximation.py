import numpy as np
from rk.runge_kutta import RungeKutta
import matplotlib.pyplot as plt

rk = RungeKutta()

t0 = 0

init_data = np.array([1., 0.])

omega = 1


def solution(t):
    A = init_data[1]
    B = init_data[0] / omega

    res = A * np.sin(omega * t) + B * np.cos(omega * t)

    return res

def f(t, x):
    res = np.zeros_like(x)
    res[0] = x[1]
    res[1] = - omega * omega * x[0]
    return res


# res_ad, t_array = rk.calc_adaptive_step(init_data=init_data, function=f, t0=t0, t_last=100, init_step=0.1,
#                                         tolerance=1e-2)


res_ad, t_array = rk.calc_fix_step(init_data=init_data, function=f, t0=t0, step=0.01, n_steps=10000)

true_solution = np.zeros(len(t_array))

for i in range(len(true_solution)):
    true_solution[i] = solution(t_array[i])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# t_array_arg = range(len(t_array))
start = 5
steps = t_array[1:] - t_array[0:-1]
print(np.mean(steps))
error = res_ad[:, 0] - true_solution
print(np.max(error))
# plt.plot(range(len(t_array) - 1), steps, label="t")
# plt.plot(t_array, res_ad[:, 0], label="res[0]")
plt.plot(t_array, res_ad[:, 0] - true_solution, label="res[0]")
plt.show()
