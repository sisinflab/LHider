import numpy as np
from scipy.sparse import csr_matrix


class RandomizeResponse:
    def __init__(self, change_probability: float, random_seed: int = 42):
        self._change_probability = change_probability
        self._seed = random_seed

    def privatize(self, input_data: csr_matrix, new_seed: int = 0) -> csr_matrix:
        seed = self._seed + new_seed
        np.random.seed(seed)
        mask = csr_matrix(np.random.choice([0, 1], p=[1 - self._change_probability, self._change_probability],
                                           size=input_data.shape))
        output_data = (input_data > mask) + (input_data < mask)
        return output_data.astype(int)
