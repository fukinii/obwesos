import numpy as np
import matplotlib.pylab as plt

from utils import solve_alpha

T_linspace = np.linspace(0.2, 15, 1000)
alpha = np.full(5, 0.1)
out = np.zeros((len(T_linspace), 6), dtype=np.double)

T = 1
n = 1e20
tolerance = 1e-10
for index, T in enumerate(T_linspace):
    alpha = solve_alpha(n=n, temperature=T, tolerance=tolerance)
    out[index, 0] = T
    out[index, 1:6] = alpha[:]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title("n = 1e17")
plt.plot(T_linspace, out[:, 1], label="alpha0")
plt.plot(T_linspace, out[:, 2], label="alpha1")
plt.plot(T_linspace, out[:, 3], label="alpha2")
plt.plot(T_linspace, out[:, 4], label="alpha3")
plt.plot(T_linspace, out[:, 5], label="alpha")
plt.xlabel("T, эВ")
plt.grid()
plt.legend()

# plt.savefig("n = 1e20.png")
plt.show()