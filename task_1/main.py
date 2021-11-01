import numpy as np
from utils import newton_method

import matplotlib.pylab as plt

I_1 = 5.98
I_2 = 18.83
I_3 = 28.4
A = 6.06e21
T = 1
n = 1e17

T_linspace = np.linspace(1e-1, 10, 100)
alpha = np.full(5, 0.1)
out = np.zeros((len(T_linspace), 6), dtype=np.double)

for index, T in enumerate(T_linspace):
    alpha = newton_method(1000, A, n, T, I_1, I_2, I_3, 1e-8)
    out[index, 0] = T
    if not (np.isnan(alpha)).any() or not (alpha == 0).any():
        out[index, 1:6] = alpha[:]


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title("n = 1e17")
plt.plot(T_linspace, out[:, 0], label="alpha0")
plt.plot(T_linspace, out[:, 1], label="alpha1")
plt.plot(T_linspace, out[:, 2], label="alpha2")
plt.plot(T_linspace, out[:, 3], label="alpha3")
plt.plot(T_linspace, out[:, 4], label="alpha")
plt.xlabel("T, эВ")
plt.grid()
plt.legend()

plt.show()
# plt.savefig("n = 1e17.png")





