from src.loader import *
from src.dataset.dataset import DPCrsMatrix
from scipy.sparse import csr_matrix
from src.recommender.neighbours_numpy import ItemKNN
from src.laplace_mechanism.mechanism import LaplaceMechanism
from src.loader.paths import RESULT_DIR
from src.dataset.generator.neighbour import NeighboursGenerator, NeighboursIterative
import os
from src.exponential_mechanism.scores import MatrixCosineSimilarity, LoadScores
import pickle

GLOBAL_SEED = 42
MAIN_DIRECTORIES = [RESULT_DIR]


def compute_recommendations(data):
    model = ItemKNN(data, k=20)
    return model.fit()


def compute_score(a, b):
    scorer = MatrixCosineSimilarity(a)
    return scorer.score_function(b)


def generate_neighbour(data, seed, modified_rating):
    generator = NeighboursIterative(random_seed=seed)
    modified_rating, neighbour = generator.generate(dataset=data, modified_ratings=modified_rating)
    pred_data = compute_recommendations(data)
    pred_neigh = compute_recommendations(neighbour)
    similarity_score = compute_score(pred_data, pred_neigh)

    return neighbour, similarity_score, modified_rating


for path in MAIN_DIRECTORIES:
    if not os.path.exists(path):
        os.makedirs(path)

# loading files
dataset_path = 'YahooMovies.tsv'
loader = TsvLoader(path=dataset_path, return_type="csr")
data = DPCrsMatrix(loader.load(), path=dataset_path)
n_users = data.n_users
n_items = data.n_items

dataset_result_dir = os.path.join(RESULT_DIR, data.name)
if not os.path.exists(dataset_result_dir):
    os.makedirs(dataset_result_dir)

print(f'data ratings: {data.transactions}')
print(f'data users: {n_users}')
print(f'data items: {n_items}')

# n_neighbours = 1
random_seed = 42
deep = 10

similarity_results = dict()
modified_ratings = []


# invece di dare una profondità della ricerca posso dare uno score target (vicino a zero)?
# è una sorta di random walk


def recursive_neighbours(data, base_seed, deep, modified_ratings):
    if deep == 0:
        return

    seed = base_seed
    neighbour, similarity_score, modified = generate_neighbour(data, seed, modified_ratings)
    # similarity_results[seed] = similarity_score non funziona il dizionario con la ricorsione
    print(similarity_score)
    print(seed)
    modified_ratings.append(modified)

    recursive_neighbours(data=neighbour, base_seed=seed + 1, deep=deep - 1, modified_ratings=modified_ratings)
    return

recursive_neighbours(data=data, base_seed=random_seed, deep=deep, modified_ratings=modified_ratings)
print(modified_ratings)
