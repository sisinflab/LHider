import random
import numpy as np
from numpy.random import laplace


class LaplaceMechanism:
    def __init__(self, eps, sensitivity, random_seed=42):

        self._random_seed = random_seed
        self._eps = eps
        self._sensitivity = sensitivity

    def privatize(self, x, n=1):
        np.random.seed(self._random_seed)
        random.seed(self._random_seed)
        return laplace(x, self._sensitivity/self._eps, n)
