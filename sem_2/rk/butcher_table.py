import numpy as np


class ButcherTable4:
    def __init__(self):
        self.order = 4
        self.n_stages = 4

        self.column = np.array([0., 0.5, 0.5, 0.])
        self.row = np.array([1. / 6, 1. / 3., 1. / 3, 1. / 6])
        self.matrix = np.zeros((4, 4))
        self.matrix[1, 0] = 0.5
        self.matrix[2, 1] = 0.5
        self.matrix[3, 2] = 1.0

