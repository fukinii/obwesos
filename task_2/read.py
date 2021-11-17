import pickle
import numpy as np
import matplotlib.pylab as plt

with open('data.pickle', 'rb') as f:
    sol = pickle.load(f)

t = np.linspace(0, 4 * 1e-4, 20000001)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title("n0 = 1e17 1/см^3, T0 = 2 эВ")

#  Настраиваем вид основных тиков:
ax.tick_params(labelsize=10, labelrotation=45)  # Поворот подписей

# #  Настраиваем вид вспомогательных тиков:
# ax.tick_params(axis='both',  # Применяем параметры к обеим осям
#                which='minor',  # Применяем параметры к вспомогательным делениям
#                direction='out',  # Рисуем деления внутри и снаружи графика
#                length=10,  # Длинна делений
#                width=2,  # Ширина делений
#                color='m',  # Цвет делений
#                pad=10,  # Расстояние между черточкой и ее подписью
#                labelsize=15,  # Размер подписи
#                labelcolor='r',  # Цвет подписи
#                bottom=True,  # Рисуем метки снизу
#                top=True,  # сверху
#                left=True,  # слева
#                right=True)  # и справа


fig.set_figwidth(12)
fig.set_figheight(8)

ax.plot(t, sol[:, 0], label="alpha0")
ax.plot(t, sol[:, 1], label="alpha1")
ax.plot(t, sol[:, 2], label="alpha2")
ax.plot(t, sol[:, 3], label="alpha3")
ax.set(xlabel='time')
plt.grid()
plt.legend()

plt.savefig("alpha(t).png")
