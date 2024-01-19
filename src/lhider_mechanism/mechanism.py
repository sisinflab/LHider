import numpy as np
from src.exponential_mechanism import *


class LHider:
    def __init__(self, randomizer, n, score, eps_exp, seed=42):
        self.seed = seed
        self.randomizer = randomizer
        self.n = n

        assert score in SCORES, f'Score type not found. Accepted scores: {SCORES.keys()}'
        self.score = score
        self._score = SCORES[score]

        self.eps_exp = eps_exp

    def privatize(self, data: np.array, seed=42):
        exponential = ExponentialMechanism(score_function=self._score(data),
                                           epsilon=self.eps_exp,
                                           random_seed=self.seed)
        randoms = [self.randomizer.privatize_np(data, relative_seed=idx) for idx in range(self.n)]
        return exponential.privatize(randoms)

    def privatize_range(self, data: np.array, seed=42):
        exponential = ExponentialMechanism(score_function=self._score(data),
                                           epsilon=self.eps_exp,
                                           random_seed=self.seed)
        randoms = [self.randomizer.privatize_np(data, relative_seed=idx) for idx in range(self.n)]
        return exponential.privatize_range(randoms)

    def outputs(self, data: np.array):
        return [self.randomizer.privatize_np(data, relative_seed=idx) for idx in range(self.n)]

    def exp_mech(self, randoms, data: np.array):
        exponential = ExponentialMechanism(score_function=self._score(data),
                                           epsilon=self.eps_exp,
                                           random_seed=self.seed)
        return exponential

    def set_exp_eps(self, eps, exp_mechanism):
        self.eps_exp = eps
        exp_mechanism.eps = eps


    def privatize_matrix(self, data: np.array):
        result = self.privatize(data[0])
        for row in data[1:]:
            result = np.concatenate([result, self.privatize(row)])
        return result

    def privatize_matrix_range(self, data: np.array):
        result = self.privatize_range(data[0])
        for row in data[1:]:
            result = np.concatenate([result, self.privatize_range(row)])
        return result

    def file_name(self, data_name: str):
        return data_name + '_' + str(self.randomizer) + '_' +\
            f'n_{self.n}_' + f'score_{self.score}_' + f'epsexp_{self.eps_exp}'
