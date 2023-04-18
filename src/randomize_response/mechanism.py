import numpy as np
from scipy.sparse import csr_matrix


class RandomizeResponse:
    def __init__(self, change_probability: float, base_seed: int = 42):
        self._change_probability = change_probability
        self._base_seed = base_seed

    def privatize(self, input_data: csr_matrix, relative_seed: int = 0) -> csr_matrix:
        seed = self._base_seed + relative_seed
        np.random.seed(seed)
        mask = csr_matrix(np.random.choice([0, 1], p=[1 - self._change_probability, self._change_probability],
                                           size=input_data.shape))
        output_data = (input_data > mask) + (input_data < mask)
        return output_data.astype(int)

    def privatize_np(self, input_data: np.matrix, relative_seed: int = 0) -> np.ndarray:
        data_seed = self._base_seed + relative_seed
        np.random.seed(data_seed)
        mask = np.random.rand(input_data.shape[0], input_data.shape[1])
        return np.logical_xor(input_data, mask > self._change_probability).astype(int)

    def privatize_choice(self, input_data: np.matrix, relative_seed: int = 0) -> np.ndarray:
        data_seed = self._base_seed + relative_seed
        np.random.seed(data_seed)
        mask = np.random.choice([0, 1], p=[1 - self._change_probability, self._change_probability],
                                size=input_data.shape)
        return np.logical_xor(input_data, mask).astype(int)
