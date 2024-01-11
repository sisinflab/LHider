import numpy as np
import math
from scipy.sparse import csr_matrix


class Generator:
    def __init__(self, base_seed: int = 42):
        self._base_seed = base_seed

    def privatize_np(self, input_data: np.array, relative_seed: int = 0) -> np.ndarray:
        pass


class RandomizeResponse(Generator):
    def __init__(self, epsilon: float, base_seed: int = 42):
        self.eps = epsilon
        self._change_probability = self.eps_to_prob()
        super().__init__(base_seed)

    def __str__(self):
        return f'randomized_response_eps{self.eps}'

    def privatize(self, input_data: csr_matrix, relative_seed: int = 0) -> csr_matrix:
        seed = self._base_seed + relative_seed
        np.random.seed(seed)
        mask = csr_matrix(np.random.choice([0, 1], p=[1 - self._change_probability, self._change_probability],
                                           size=input_data.shape))
        output_data = (input_data > mask) + (input_data < mask)
        return output_data.astype(int)

    def privatize_np(self, input_data: np.array, relative_seed: int = 0) -> np.ndarray:
        data_seed = self._base_seed + relative_seed
        np.random.seed(data_seed)
        mask = np.random.rand(*input_data.shape)
        return np.logical_xor(input_data, mask < self._change_probability).astype(int)

    def privatize_array(self, input_data: np.array, relative_seed: int = 0) -> np.ndarray:
        data_seed = self._base_seed + relative_seed
        np.random.seed(data_seed)
        mask = np.random.rand(input_data.shape[0], input_data.shape[1])
        return np.logical_xor(input_data, mask < self._change_probability).astype(int)

    def privatize_choice(self, input_data: np.array, relative_seed: int = 0) -> np.ndarray:
        data_seed = self._base_seed + relative_seed
        np.random.seed(data_seed)
        mask = np.random.choice([0, 1], p=[1 - self._change_probability, self._change_probability],
                                size=input_data.shape)
        return np.logical_xor(input_data, mask).astype(int)

    def eps_to_prob(self, eps: float = None):
        if eps is None:
            eps = self.eps
        return 1 / (1 + math.exp(eps))

    def prob_to_eps(self, prob: float = None):
        if prob is None:
            prob = self._change_probability
        return math.log(max((prob / (1 - prob)), ((1 - prob) / prob)))


class RandomGenerator(Generator):
    def __init__(self, base_seed: int = 42):
        super().__init__(base_seed)

    def privatize_np(self, input_data: np.array, relative_seed: int = 0) -> np.ndarray:
        data_seed = self._base_seed + relative_seed
        np.random.seed(data_seed)
        return np.random.randint(0, 2, size=input_data.shape)
