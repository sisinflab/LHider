import numpy as np


class ExponentialMechanism:
    def __init__(self, score_function, epsilon=0.5, random_seed=42):

        self._random_seed = random_seed
        np.random.seed(self._random_seed)
        self.score_function = score_function
        self.sensitivity = self.score_function.sensitivity
        self.eps = epsilon

    def scores(self, output):
        return np.array([self.score_function(x) for x in output])

    def probabilities(self, output):
        exponent = (self.eps * self.scores(output)) / (2 * self.sensitivity)
        exponent = exponent.astype('float128')
        probabilities = np.exp(exponent)
        probabilities = probabilities / np.sum(probabilities)
        return probabilities

    def privatize(self, output, probs=None):
        print(f'{self.__class__.__name__}: choosing between {len(output)} possible outputs')
        if probs is None:
            probs = self.probabilities(output)
        assert len(probs) == len(output),\
            f'{self.__class__.__name__}: output and probabilities must have the same length'
        return np.random.choice(output, p=probs)
