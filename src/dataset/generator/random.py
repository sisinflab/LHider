import random
import numpy as np
import scipy


class DiFazioGenerator:
    def __init__(self, users, items, random_seed):
        self.seed = random_seed
        self._n_users = users
        self._n_items = items
        random.seed(self.seed)
        np.random.seed(self.seed)

    def generate(self, n_ratings, new_seed=0):
        seed = self.seed + new_seed
        random.seed(seed)
        np.random.seed(seed)
        density = n_ratings / (self._n_users * self._n_items)
        return scipy.sparse.csr_array(
            scipy.sparse.random(self._n_users, self._n_items, density, random_state=seed, dtype=bool))
