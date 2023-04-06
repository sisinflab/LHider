import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from math import prod, isnan
import numpy as np
import scipy
import pickle
import os


class ScoreFunction:
    def __init__(self, data):
        self.sensitivity = None
        self.data = data

    def score_function(self, x):
        pass

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


class MatrixCosineSimilarity(ScoreFunction):
    def __init__(self, data):
        self.sensitivity = 1
        super(MatrixCosineSimilarity, self).__init__(data)

    def score_function(self, x):
        return np.mean(np.sum(self.data * x, axis=1)
                       / (np.sum(self.data*self.data, axis=1) ** .5 * np.sum((x * x), axis=1) ** .5))


class LoadScores(ScoreFunction):

    def __init__(self, path, sensitivity, dropna=True):
        assert os.path.exists(path)
        f'Scores found at: \'{path}\''
        with open(path, 'rb') as file:
            data = pickle.load(file)

        if dropna:
            data = {k: v for k, v in data.items() if not (isnan(v))}

        assert isinstance(data, dict)
        super(LoadScores, self).__init__(data)
        self.sensitivity = sensitivity

    def score_function(self, x):
        assert x in self.data
        return self.data[x]
