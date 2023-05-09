from src.loader.paths import *
from src.exponential_mechanism.scores import LoadScores
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter


def score_plot(scores: list, decimal=4):
    scores = [round(s, decimal) for s in scores]
    count_scores = Counter(scores)
    ordered_scores = sorted(scores)
    count_ordered_scores = {k: count_scores[k] for k in ordered_scores}
    plt.plot(count_ordered_scores.keys(), count_ordered_scores.values())
    plt.show()

dataset_name = 'facebook_books'
eps_rr = 3.0

for eps_rr in [1.0, 3.0, 6.0]:
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

    print('----')
    print(eps_rr)
    print(max_score)
    print(min_score)
    print(mean_score)
    print(std_dev)
    print('----')


    score_plot(possible_output_scores, decimal=4)
    print()
