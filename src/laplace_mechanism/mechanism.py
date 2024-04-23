import random
import numpy as np
import math
from numpy.random import laplace
from scipy.stats import dlaplace


class DiscreteLaplaceMechanismScipy:
    def __init__(self, eps, sensitivity, min_val, max_val, base_seed=42):
        self._eps = eps
        self._sensitivity = sensitivity
        self._min_val = min_val
        self._max_val = max_val
        self._base_seed = base_seed
        self._a = self._eps/self._sensitivity

    def privatize(self):
        pass

    def privatize_np(self, input_data: np.array, relative_seed: int = 0) -> np.ndarray:
        data_seed = self._base_seed + relative_seed
        np.random.seed(data_seed)
        noise = np.array([dlaplace.rvs(self._a) for _ in range(input_data.shape[0] * input_data.shape[1])]).reshape(input_data.shape)
        return np.clip(input_data + noise, self._min_val, self._max_val)



class LaplaceMechanism:
    def __init__(self, eps, sensitivity, random_seed=42):

        self._random_seed = random_seed
        self._eps = eps
        self._sensitivity = sensitivity

    def privatize(self, x, n=1):
        np.random.seed(self._random_seed)
        random.seed(self._random_seed)
        return laplace(x, self._sensitivity/self._eps, n)


class DiscreteLaplaceMechanism:
    def __init__(self, eps, sensitivity, min_val, max_val, base_seed=42):
        self._eps = eps
        self._sensitivity = sensitivity
        self._min_val = min_val
        self._max_val = max_val
        self._base_seed = base_seed
        self._a = self._eps/self._sensitivity
        self._samples = np.array(range(-500000, 500001))
        self.probs = np.array([self.p(x) for x in self._samples])
        total = sum(self.probs)
        diff = 1-total
        if diff > 0.01:
            raise ValueError
        self.probs[np.where(self._samples == 0)[0][0]] += diff


    def p(self, x):
        xv = abs(x)
        ea = math.exp(-self._a)
        a = (1 - ea) / (1 + ea)
        b = math.exp(-self._a * xv)
        return a * b

    def privatize(self):
        pass

    def privatize_np(self, input_data: np.array, relative_seed: int = 0) -> np.ndarray:
        data_seed = self._base_seed + relative_seed
        np.random.seed(data_seed)
        print('qui')
        noise = np.random.choice(self._samples, p=self.probs, size=input_data.shape)
        print('fatto')
        return np.clip(input_data + noise, self._min_val, self._max_val)
