from src.loader import *
from src.dataset.dataset import DPCrsMatrix
from scipy.sparse import csr_matrix
from src.recommender.neighbours_numpy import ItemKNN
from src.randomize_response.mechanism import RandomizeResponse
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


def gen_and_score(ratings, change_probability, seed=0):
    ratings_generator = RandomizeResponse(change_probability=change_probability, base_seed=seed)
    generated_dataset = ratings_generator.privatize(data.values)
    generated_dataset = DPCrsMatrix(generated_dataset)
    generated_ratings = compute_recommendations(generated_dataset)
    score = compute_score(ratings, generated_ratings)
    return score, seed

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

random_seed = 42
change_probability = 0.002

# recommendations
print(f'\nComputing recommendations')
ratings = compute_recommendations(data)
score, seed = gen_and_score(ratings, 0, seed=0)
print(f'identity score: {score}')

# score
score, seed = gen_and_score(ratings, change_probability, seed=0)
print(f'score: {score}')
