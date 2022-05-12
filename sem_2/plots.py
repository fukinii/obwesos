import sys

# sys.path.extend(['/Users/u19800196/ilya/obwesos/sem_2'])
sys.path.extend(['/home/fukin/obwesos/obwesos/sem_2'])
import pickle

import numpy as np
from rk.runge_kutta import RungeKutta
import matplotlib.pyplot as plt
# from model.model import PhotoionizationModel
from model.model2 import PhotoionizationModel
from model.consts import *
from colour import Color

n = 425.4
r_array = np.linspace(R_sun_non_dim, r_non_dim, 10)
# r_array = np.linspace(r_non_dim, r_non_dim, 1)
model = PhotoionizationModel(n, r_array=r_array)

path = "/home/fukin/obwesos/obwesos/sem_2/output.pickle"
with open(path, "rb") as f:
    data = pickle.load(f)

solution = data[0]
t_array = data[1]
j_array = data[2]
current_time = data[3]
step = data[4]
iteration_number = int(data[5])

# print(iteration_number)
# print(len(t_array))
# iteration_number = 1000
a = 1

fig = plt.figure(figsize=(14, 10))
plot_ax_1 = fig.add_subplot(2, 2, 1)
plot_ax_2 = fig.add_subplot(2, 2, 2)
plot_ax_3 = fig.add_subplot(2, 2, 3)
plot_ax_4 = fig.add_subplot(2, 2, 4)

alpha = solution[:, :, 0]
T_e = solution[:, :, 1]
T = solution[:, :, 2]

# t_linspace = np.linspace(0, iteration_number, iteration_number + 1)

t_linspace_sparced = []
alpha_sparced = []
T_sparced = []
T_e_sparced = []
j_array_sparced = []

# for i in range(iteration_number):

result_t = []
print(t_array.shape)
for t_i, t in enumerate(t_array[:10000]):
    if t_i % 10 == 0:
        result_t.append([t_i, t])

for r_i in range(len(r_array)):
    alpha_i = []
    T_i = []
    T_e_i = []
    j_i = []
    for t_i, t in result_t:
        alpha_i.append(np.log10(alpha[t_i, r_i]))
        T_i.append(np.log10(T[t_i, r_i]))
        T_e_i.append(np.log10(T_e[t_i, r_i]))

        j_i.append(np.log10(j_array[t_i, r_i]))

    j_array_sparced.append(j_i)
    alpha_sparced.append(alpha_i)
    T_sparced.append(T_i)
    T_e_sparced.append(T_e_i)

alpha_sparced = np.array(alpha_sparced)
T_sparced = np.array(T_sparced)
T_e_sparced = np.array(T_e_sparced)
j_array_sparced = np.array(j_array_sparced)

print(r_array.shape)
print(alpha_sparced.shape)

red = Color("red")
colors = list(red.range_to(Color("green"), len(result_t)))

for i in range(len(result_t)):
    plot_ax_1.plot(r_array,
                   alpha_sparced[:, i],
                   # label=f'r: {t}%.2f',
                   color=colors[i].hex)
    # plot_ax_1.plot(t_linspace_sparced, alpha_sparced[:, r_i], label=f'r: {r_array[r_i]}%.2f')
# plot_ax_1.set_title('alpha(t)', fontsize=15)
# plot_ax_1.set_xlabel(r'$log10(t)$', loc='right')
# plot_ax_1.set_ylabel(r'$log10(alpha)$')
# plot_ax_1.legend()
plot_ax_1.grid()

for i in range(len(result_t)):
    plot_ax_2.plot(r_array,
                   T_e_sparced[:, i],
                   # label=f'r: {t}%.2f',
                   color=colors[i].hex)
# plot_ax_2.set_title('T(t)', fontsize=15)
# plot_ax_2.set_xlabel(r'$log10(t)$', loc='right')
# plot_ax_2.set_ylabel(r'$log10(T)$')
plot_ax_2.grid()

for i in range(len(result_t)):
    plot_ax_3.plot(r_array,
                   T_sparced[:, i],
                   # label=f'r: {t}%.2f',
                   color=colors[i].hex)
# plot_ax_3.legend()
# plot_ax_3.set_title('T_e(t)', fontsize=15)
# plot_ax_3.set_xlabel(r'$log10(t)$', loc='right')
# plot_ax_3.set_ylabel(r'$log10(T_e)$')
plot_ax_3.grid()

for i in range(len(result_t)):
    plot_ax_4.plot(r_array,
                   j_array_sparced[:, i],
                   # label=f'r: {t}%.2f',
                   color=colors[i].hex)
# plot_ax_4.legend()
# plot_ax_4.set_title('j_v(t)', fontsize=15)
# plot_ax_4.set_xlabel(r'$log10(t)$', loc='right')
# plot_ax_4.set_ylabel(r'$log10(j_v)$')
plot_ax_4.grid()

# plt.savefig('Графики_result_r.png')
plt.show()

# solution = data[0]
# t_array = data[1]
# j_array = data[2]
# current_time = data[3]
# step = data[4]
# iteration_number = int(data[5])
#
# print(iteration_number)
# print(len(t_array))
# # iteration_number = 1000
# a = 1
#
# fig = plt.figure(figsize=(14, 10))
# plot_ax_1 = fig.add_subplot(2, 2, 1)
# plot_ax_2 = fig.add_subplot(2, 2, 2)
# plot_ax_3 = fig.add_subplot(2, 2, 3)
# plot_ax_4 = fig.add_subplot(2, 2, 4)
#
# alpha = solution[:, :, 0]
# T_e = solution[:, :, 1]
# T = solution[:, :, 2]
#
# # t_linspace = np.linspace(0, iteration_number, iteration_number + 1)
#
# t_linspace_sparced = []
# alpha_sparced = []
# T_sparced = []
# T_e_sparced = []
# j_array_sparced = []
#
# # for i in range(iteration_number):
# for t_i, t in enumerate(t_array):
#
#     # print(t_i, t)
#     if t_i % 10 == 0:
#         alpha_i = []
#         T_i = []
#         T_e_i = []
#         j_i = []
#         for r_i in range(len(r_array)):
#             # alpha_i.append(alpha[t_i, r_i])
#             # T_i.append(T[t_i, r_i])
#             # T_e_i.append(T_e[t_i, r_i])
#
#             alpha_i.append(np.log10(alpha[t_i, r_i]))
#             T_i.append(np.log10(T[t_i, r_i]))
#             T_e_i.append(np.log10(T_e[t_i, r_i]))
#
#             j_i.append(np.log10(j_array[t_i, r_i]))
#
#         j_array_sparced.append(j_i)
#         alpha_sparced.append(alpha_i)
#         T_sparced.append(T_i)
#         T_e_sparced.append(T_e_i)
#
#         # alpha_0_sparced.append(alpha[i, 0])
#         # alpha_4_sparced.append(alpha[i, 4])
#         # alpha_9_sparced.append(alpha[i, 9])
#
#         # T_0_sparced.append(T[i, 0])
#         # T_4_sparced.append(T[i, 4])
#         # T_9_sparced.append(T[i, 9])
#         #
#         # T_e_0_sparced.append(T_e[i, 0])
#         # T_e_4_sparced.append(T_e[i, 4])
#         # T_e_9_sparced.append(T_e[i, 9])
#         # t_linspace_sparced.append(t)
#         # print(t, np.log10(t))
#         t_linspace_sparced.append(np.log10(t))
#
#     # i += 1
#
# alpha_sparced = np.array(alpha_sparced)
# T_sparced = np.array(T_sparced)
# T_e_sparced = np.array(T_e_sparced)
# j_array_sparced = np.array(j_array_sparced)
#
# for r_i in range(len(r_array)):
#     plot_ax_1.plot(t_linspace_sparced, alpha_sparced[:, r_i], label="{0:.2E}".format(r_array[r_i]))
#     # plot_ax_1.plot(t_linspace_sparced, alpha_sparced[:, r_i], label=f'r: {r_array[r_i]}%.2f')
# plot_ax_1.set_title('alpha(t)', fontsize=15)
# plot_ax_1.set_xlabel(r'$log10(t)$', loc='right')
# plot_ax_1.set_ylabel(r'$log10(alpha)$')
# plot_ax_1.legend()
# plot_ax_1.grid()
#
# for r_i in range(len(r_array)):
#     plot_ax_2.plot(t_linspace_sparced, T_sparced[:, r_i])
# plot_ax_2.set_title('T(t)', fontsize=15)
# plot_ax_2.set_xlabel(r'$log10(t)$', loc='right')
# plot_ax_2.set_ylabel(r'$log10(T)$')
# plot_ax_2.grid()
#
# for r_i in range(len(r_array)):
#     plot_ax_3.plot(t_linspace_sparced, T_e_sparced[:, r_i])
# # plot_ax_3.legend()
# plot_ax_3.set_title('T_e(t)', fontsize=15)
# plot_ax_3.set_xlabel(r'$log10(t)$', loc='right')
# plot_ax_3.set_ylabel(r'$log10(T_e)$')
# plot_ax_3.grid()
#
# for r_i in range(len(r_array)):
#     plot_ax_4.plot(t_linspace_sparced, j_array_sparced[:, r_i])
# # plot_ax_4.legend()
# plot_ax_4.set_title('j_v(t)', fontsize=15)
# plot_ax_4.set_xlabel(r'$log10(t)$', loc='right')
# plot_ax_4.set_ylabel(r'$log10(j_v)$')
# plot_ax_4.grid()
#
# plt.savefig('Графики_result.png')
# plt.show()
