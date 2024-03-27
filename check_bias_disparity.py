import os

import pandas as pd

original_data = pd.read_csv('data/yahoo_movies/train.tsv', sep='\t', names=['u', 'i', 'r'])
perturbed_data = pd.read_csv('/Users/sciueferrara/Downloads/100/for_bias/randomized_8_10_0.001_126_8.0005000139.tsv', sep='\t', names=['u', 'i', 'r'])

original_items_quantiles = pd.qcut(original_data.groupby('i').size(), [0, 0.25, 0.7, 1], labels=[1, 2, 3])
# perturbed_items_quantiles = pd.qcut(perturbed_data.groupby('.size(), 4, labels=[1, 2, 3, 4])

original_data = original_data.merge(original_items_quantiles.to_frame(), on='i')
perturbed_data = perturbed_data.merge(original_items_quantiles.to_frame(), on='i')
# perturbed_items_quantiles = pd.qcut(perturbed_data.groupby('i').size(), 4, labels=[1, 2, 3, 4])

n_transactions_original = len(original_data)
n_transactions_perturbed = len(perturbed_data)
n_items = original_data['i'].nunique()

bd = []
for i in range(1, 4):
    n_transactions_c_original = len(original_data[original_data[0] == i])
    n_transactions_c_perturbed = len(perturbed_data[perturbed_data[0] == i])
    pr_c_original = n_transactions_c_original / n_transactions_original
    # n_items_c = original_data[original_data[0] == i]['i'].nunique()
    pr_c_perturbed = n_transactions_c_perturbed / n_transactions_perturbed
    # p_c = n_items_c / n_items
    bd_c = pr_c_perturbed / pr_c_original - 1
    bd.append(bd_c)

print(bd)

print('faccio similarità')

most_pop_original = original_data.groupby('i').size().sort_values().index[-1]

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

original_matrix = csr_matrix((original_data['r'], (original_data['u'], original_data['i'])), shape=(original_data['u'].nunique(), original_data['i'].nunique()))
original_similarity = cosine_similarity(original_matrix.T)
original_most_similar = original_similarity[most_pop_original].argsort()[-4:][::-1]
print(original_most_similar)

path_perturbed = '/Users/sciueferrara/Downloads/100/for_bias'
for data in os.listdir(path_perturbed):
    if data != '.DS_Store':
        print(data)
        perturbed_data = pd.read_csv(os.path.join(path_perturbed, data), sep='\t', names=['u', 'i'])
        perturbed_matrix = csr_matrix((np.ones(len(perturbed_data)), (perturbed_data['u'], perturbed_data['i'])),
                                     shape=(perturbed_data['u'].nunique(), perturbed_data['i'].nunique()))
        perturbed_similarity = cosine_similarity(perturbed_matrix.T)
        perturbed_most_similar = perturbed_similarity[most_pop_original].argsort()[-4:][::-1]
        print(perturbed_most_similar)

