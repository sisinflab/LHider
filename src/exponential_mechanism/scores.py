import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from math import isnan
import numpy as np
import scipy
import pickle
import os
import pandas as pd


class ScoreFunction:
    def __init__(self, data):
        self.sensitivity = None
        self.data = data
        self.max = None

    def score_function(self, x):
        pass

    def normalize(self, score):
        assert self.max, f'max value is not defined for {self.__class__.__name__} class'
        return score / self.max

    def __call__(self, x):
        return self.score_function(x)


class Count(ScoreFunction):

    def __init__(self, data):
        super().__init__(data)
        self.sensitivity = 1

    def score_function(self, column):
        return sum(self.data[column])


class ItemSimilarity(ScoreFunction):

    def __init__(self, data, similarity=cosine_similarity, sensitivity=1):
        super(ItemSimilarity, self).__init__(data)

        self.similarity_fun = similarity
        self.sensitivity = sensitivity
        self.similarity_matrix = self.similarity_fun(data.values.T)

    def score_function(self, permutation_matrix):
        users_arrays = ((self.data.user_items(u), permutation_matrix[u]) for u in range(self.data.users))
        return sum(map(self.user_similarity, tqdm.tqdm(users_arrays, total=self.data.users)))/self.data.users

    def user_similarity(self, arrays):
        sub_matrix = self.similarity_matrix[arrays[0]][:, arrays[1]]
        # result = self.hungarian_algorithm(sub_matrix) / sub_matrix.shape[0]
        result = self.hungarian_algorithm(sub_matrix)
        return result

    @staticmethod
    def hungarian_algorithm(matrix):
        max_indices = scipy.optimize.linear_sum_assignment(matrix, maximize=True)
        result = np.sum(matrix[max_indices[0], max_indices[1]])
        return result


class Distance(ScoreFunction):
    def __init__(self, data, similarity=cosine_similarity, sensitivity=1):
        super(Distance, self).__init__(data)

        self.similarity_fun = similarity
        self.sensitivity = sensitivity
        self.similarity_matrix = self.similarity_fun(data.values.T)

    def score_function(self, generated_matrix):
        generated_matrix_similarity = self.similarity_fun(generated_matrix.T)
        users_arrays = ((sum(self.similarity_matrix[self.data.user_items(u)]),
                         sum(generated_matrix_similarity[generated_matrix[[u], :].indices]))
                        for u in range(self.data.n_users))
        return sum(map(self.user_similarity, tqdm.tqdm(users_arrays, total=self.data.users))) / self.data.n_users

    def user_similarity(self, arrays):
        return cosine_similarity(arrays[0].reshape(1, -1), arrays[1].reshape(1, -1))[0][0]


class MatrixUserCosineSimilarity(ScoreFunction):
    def __init__(self, data):
        super(MatrixUserCosineSimilarity, self).__init__(data)
        self.sensitivity = 1

    def score_function(self, x):
        return np.mean(np.sum(self.data * x, axis=1)
                       / (np.sum(self.data*self.data, axis=1) ** .5 * np.sum((x * x), axis=1) ** .5))


class MatrixItemCosineSimilarity(ScoreFunction):
    def __init__(self, data):
        super(MatrixItemCosineSimilarity, self).__init__(data)
        self.sensitivity = 1

    def score_function(self, x):
        return np.mean(np.sum(self.data.T * x.T, axis=1)
                       / (np.sum(self.data.T * self.data.T, axis=1) ** .5 * np.sum((x.T * x.T), axis=1) ** .5))


class ManhattanDistance(ScoreFunction):
    def __init__(self, data):
        super(ManhattanDistance, self).__init__(data)
        self.sensitivity = 1 / self.data.size
        self.max = self.data.size
        self.range = 1

    def __str__(self):
        return 'manhattan_distance'

    def score_function(self, x):
        scores = np.sum(np.abs(self.data - x))
        normalized_scores = self.normalize(scores)
        return 1 - normalized_scores

class JaccardDistance(ScoreFunction):
    def __init__(self, data):
        super(JaccardDistance, self).__init__(data)
        self.sensitivity = 1 / (np.sum(self.data == 1) - 1)
        self.max = 1
        self.range = 1

    def __str__(self):
        return 'jaccard_distance'

    def score_function(self, x):
        intersection = np.sum(np.logical_and((self.data == 1), (x == 1)))
        return intersection / (intersection + np.sum(self.data != x))


class MatrixManhattanDistance(ScoreFunction):
    def __init__(self, data):
        super(MatrixManhattanDistance, self).__init__(data)
        self.sensitivity = 1 / self.data.size
        self.max = self.data.size

    def score_function(self, x):
        scores = np.sum(np.abs(self.data - x))
        normalized_scores = self.normalize(scores)
        return 1 - normalized_scores


class MatrixEuclideanDistance(ScoreFunction):
    def __init__(self, data):
        super(MatrixEuclideanDistance, self).__init__(data)
        self.sensitivity = 1 / np.sqrt(self.data.size)
        self.max = np.sqrt(self.data.size)

    def score_function(self, x):
        scores = np.sqrt(np.sum(np.power(self.data - x, 2)))
        normalized_scores = self.normalize(scores)
        return 1 - normalized_scores


class MatrixJaccardDistance(ScoreFunction):
    def __init__(self, data: np.array):
        super(MatrixJaccardDistance, self).__init__(data)
        self.sensitivity = 1 / (np.sum(self.data == 1) - 1)

    def score_function(self, x: np.array):
        intersection = np.sum(np.logical_and((self.data == 1), (x == 1)))
        return intersection / (intersection + np.sum(self.data != x))


class LoadScores(ScoreFunction):

    def __init__(self, path, sensitivity, dropna=True):
        # load scores from file
        if not os.path.exists(path):
            raise FileNotFoundError(f'Scores file not found at \'{path}\'. Please, check your files.')
        with open(path, 'rb') as file:
            data = pickle.load(file)
        print(f'Scores found at: \'{path}\'')

        if dropna:
            # TODO: uniformare il tipo di struttura dati usata per salvare gli score
            if type(data.values) == int:
                print("int")
                data = {k: v for k, v in data.items() if not (isnan(v))}
            elif type(data.values) == dict:
                print("dict")
                data = {k: v['scores'] for k, v in data.items() if not (isnan(v['score']))}

        assert isinstance(data, dict)

        super(LoadScores, self).__init__(data)
        self.sensitivity = sensitivity

    def score_function(self, x):
        assert x in self.data, f'Sample {x} not found in data.'
        score = self.data[x]
        # TODO: uniformare il tipo di struttura dati usata per salvare gli score
        if type(score) == dict:
            return score['score']
        else:
            return score


class Scores:

    def __init__(self, path):
        assert os.path.exists(path)
        self.path = os.path.abspath(path)
        print(f'Scores found at: \'{self.path}\'')
        self.data = None

    def load(self, dropna=True):
        print(f'Loading scores from: \'{self.path}\'')
        with open(self.path, 'rb') as file:
            data = pickle.load(file)

        if dropna:
            data = self.drop_na(data)

        assert isinstance(data, dict)
        self.data = data

    def drop_na(self, data: dict):
        return {k: v for k, v in data.items() if not (isnan(v))}

    def to_dataframe(self):
        data = pd.DataFrame()
        data['id'] = self.data.keys()
        data['scores'] = self.data.values()
        return data

    def decimal(self, decimals):
        self.data = {k: round(v, decimals) for k, v in self.data.items()}


class Score:

    def __init__(self, scores: dict,
                 dataset_name=None,
                 dataset_type=None,
                 score_type=None,
                 eps=None,
                 generations=None):
        assert len(scores) > 0, "Scores not found."

        self._scores: dict = scores
        self._dataset_name = str(dataset_name)
        self._dataset_type = str(dataset_type)
        self._score_type = str(score_type)
        self._eps = str(eps)
        if generations is None:
            generations = len(scores)
        assert generations <= len(scores)
        self._gen = generations

        selected_keys = list(scores.keys())[:self._gen]
        self._scores = {k: self._scores[k] for k in selected_keys}

        self._data: np.ndarray = np.array(list(self._scores.values()))

    def __len__(self):
        return len(self._data)

    def drop_na(self):
        data = self._scores.copy()
        return {k: v for k, v in data.items() if not (isnan(v))}

    def to_dataframe(self):
        data = pd.DataFrame()
        data['id'] = self._scores.keys()
        data['scores'] = self._scores.values()
        return data

    def approx(self, decimals):
        self._scores = {k: round(v, decimals) for k, v in self._scores.items()}

    def mean(self):
        """
        @return: mean value of the scores
        """
        if len(self._data) == 0:
            return 0
        else:
            return self._data.mean()
    def max(self):
        """
        @return: max value of the scores
        """
        if len(self._data) == 0:
            return 0
        else:
            return self._data.max()

    def min(self):
        """
        @return: min value of the scores
        """
        if len(self._data) == 0:
            return 0
        else:
            return self._data.min()

    def std(self):
        """
        @return: standard deviation of the scores
        """
        if len(self._data) == 0:
            return 0
        else:
            return self._data.std()

    @property
    def data(self):
        """
        @return: the numpy array containing the scores
        """
        return self._data

    def score_name(self, decimal:int =None):
        name = ''
        if self._dataset_name:
            name += self._dataset_name
        if self._dataset_type:
            name += '_' + self._dataset_type
        if self._score_type:
            name += '_' + self._score_type
        if self._eps:
            name += '_' + self._eps
        if decimal:
            name += '_' + str(decimal)
        return name

    def values_over_threshold(self, thresh):
        return sum(self._data > thresh)


SCORERS = {
    'manhattan': MatrixManhattanDistance,
    'euclidean': MatrixEuclideanDistance,
    'cosineUser': MatrixUserCosineSimilarity,
    'cosineItem': MatrixItemCosineSimilarity,
    'jaccard': MatrixJaccardDistance
}
