import numpy as np
import matplotlib.pyplot as plt
from integral.integral import integrate

def func(x):
    return x * x


x_linspace = np.linspace(0., 1., 101)
x_linspace_sin = np.linspace(0., np.pi, 101)


integral = integrate(func=func, x_linspace=x_linspace)
integral_sin = integrate(func=np.sin, x_linspace=x_linspace_sin)
a = 1