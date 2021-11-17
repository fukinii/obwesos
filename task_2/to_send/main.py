import numpy as np
import matplotlib.pyplot as plt
from utils import rungekutta4, calc_func

t = np.linspace(0, 4 * 1e-4, 20000001)

# Из задания 6 для n = 1e17, T = 2
alpha_0 = np.array([2.59333106e-05, 0.11553579, 0.83412872, 0.05030956])

sol = rungekutta4(calc_func, alpha_0, t)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title("n0 = 1e17 1/см^3, T0 = 2 эВ")

ax.tick_params(labelsize=10, labelrotation=45)

fig.set_figwidth(12)
fig.set_figheight(8)

ax.plot(t, sol[:, 0], label="alpha0")
ax.plot(t, sol[:, 1], label="alpha1")
ax.plot(t, sol[:, 2], label="alpha2")
ax.plot(t, sol[:, 3], label="alpha3")
ax.set(xlabel='time')
plt.grid()
plt.legend()

# plt.savefig("alpha(t).png")