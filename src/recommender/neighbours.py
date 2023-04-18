import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ItemKNNDense:

    def __init__(self, data: np.array, k, similarity_fun=cosine_similarity):
        self.data = data
        self.shape = data.shape
        self.n_items = self.shape[1]
        self._similarity_fun = similarity_fun
        self._similarity_matrix = self._similarity_fun(np.transpose(data))
        self._k = k
        self._top_k = None

    @property
    def top_k(self):
        if self._top_k is None:
            self._top_k = self.compute_top_k(self._k)
        return self._top_k

    def compute_top_k(self, k):
        return {i: np.argpartition(-self._similarity_matrix[i, np.r_[:i, i + 1:self.n_items]], k)[:k] for i in
                range(self.n_items)}

    def fit(self):
        predictions = np.zeros(self.shape)
        for i in range(self.n_items):
            topk_sim = self._similarity_matrix[self.top_k[i], i]
            ratings_sim = np.multiply(self.data[:, self.top_k[i]], topk_sim)
            predictions[:, i] = ratings_sim.mean(axis=1).reshape(-1)
        predictions[self.data == 1] = 1
        return predictions


class ItemKNNSparse:

    def __init__(self, data, k, similarity_fun=cosine_similarity):
        self._data = data
        self._similarity_fun = similarity_fun
        self._similarity_matrix = self._similarity_fun(data.values.T)
        self._k = k
        self._top_k = None

    @property
    def top_k(self):
        if self._top_k is None:
            self._top_k = self.compute_top_k(self._k)
        return self._top_k

    def compute_top_k(self, k):
        return {i: np.argpartition(-self._similarity_matrix[i, np.r_[:i, i + 1:self._data.n_items]], k)[:k] for i in
                range(self._data.n_items)}

    def fit(self):
        predictions = np.zeros(self._data.shape)
        for i in range(self._data.n_items):
            topk_sim = self._similarity_matrix[self.top_k[i], i]
            ratings_sim = self._data[:, self.top_k[i]].multiply(topk_sim)
            predictions[:, i] = ratings_sim.mean(axis=1).reshape(-1)
        predictions[(self._data.dataset == 1).todense()] = 1
        return predictions
