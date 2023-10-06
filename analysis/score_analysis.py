from src.loader.paths import *
from src.exponential_mechanism.scores import LoadScores
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import pandas as pd
from src.dataset.dataset import DPDataFrame
from src.loader.loaders import *
from src.exponential_mechanism.scores import *
from src.exponential_mechanism.scores import Scores


def score_to_dataframe(scores: list, decimal=4):
    scores = [round(s, decimal) for s in scores]
    count_scores = Counter(scores)
    ordered_scores = sorted(scores)
    count_ordered_scores = {k: count_scores[k] for k in ordered_scores}
    return pd.DataFrame(data=count_ordered_scores.items(), columns=['score', 'freq'])

def score_plot(scores: list, decimal=4):
    scores = [round(s, decimal) for s in scores]
    count_scores = Counter(scores)
    ordered_scores = sorted(scores)
    count_ordered_scores = {k: count_scores[k] for k in ordered_scores}
    plt.plot(count_ordered_scores.keys(), count_ordered_scores.values())
    plt.show()

#
# dataset_name = 'yahoo_movies'
# eps_rr = 3.0
#
# for eps_rr in [1.0, 3.0, 6.0]:
#     scores_dir = scores_directory(dataset_dir=dataset_directory(dataset_name),
#                                   eps=eps_rr)
#
#     seed_scores = LoadScores(path=scores_file_path(scores_dir),
#                              sensitivity=1)
#     possible_output_seeds = list(seed_scores.data.keys())
#     possible_output_scores_temp = list(seed_scores.data.values())
#
#     possible_output_scores = [el['score'] if el['score'] else el for el in possible_output_scores_temp]
#
#     max_score = np.max(possible_output_scores)
#     min_score = np.min(possible_output_scores)
#     mean_score = np.mean(possible_output_scores)
#     std_dev = np.std(possible_output_scores)
#
#     print('----')
#     print(eps_rr)
#     print(max_score)
#     print(min_score)
#     print(mean_score)
#     print(std_dev)
#     print('----')
#
#
#     # score_plot(possible_output_scores, decimal=4)
#     path = os.path.join("../results", dataset_name, "scores", f"eps_{eps_rr}")
#     os.makedirs(path, exist_ok=True)
#
#     score_to_dataframe(possible_output_scores, decimal=4).to_csv(os.path.join(path, "score_frequency.tsv"), sep="\t", index=False)
#     print()


dataset_name = 'facebook_books'
datatype = 'clean'
score = 'euclidean'
eps = '1.0'


dataset_path = dataset_filepath(dataset_name, datatype)

from src.loader.loaders import TsvLoader
loader = TsvLoader(dataset_path)
data = loader.load()

dataset = DPDataFrame(data)

print(dataset.n_users)
print(dataset.n_items)
print(dataset.transactions)

score_loader = ScoreLoader(dataset_name=dataset_name,
                           dataset_type=datatype,
                           score_type=score,
                           eps=eps)
score_values = score_loader.load()
print(len(score_values))

scores = Score(score_values)

print()
