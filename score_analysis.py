import os
from src.loader.paths import *
from src.exponential_mechanism.scores import LoadScores
import numpy as np


dataset_name = 'facebook_books'
eps_rr = 1.0

scores_dir = scores_directory(dataset_dir=dataset_directory(dataset_name),
                              eps=eps_rr)

seed_scores = LoadScores(path=scores_file_path(scores_dir),
                         sensitivity=1)
possible_output_seeds = list(seed_scores.data.keys())
possible_output_scores = list(seed_scores.data.values())

max_score = np.max(possible_output_scores)
min_score = np.min(possible_output_scores)
mean_score = np.mean(possible_output_scores)
std_dev = np.std(possible_output_scores)


print()