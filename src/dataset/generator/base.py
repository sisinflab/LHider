import random
import numpy as np


class Generator:

    def __init__(self, random_seed=42):
        self._seed = random_seed

    def set_seed(self):
        random.seed(self._seed)
        np.random.seed(self._seed)

    def generate(self, *args, **kwargs):
        pass
