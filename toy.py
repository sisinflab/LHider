import random

import pandas as pd
from src.exponential_mechanism import *
from src.randomize_response import *
import itertools

TRIALS = 10
N = 64


def score_stats(method, scores):
    print('*** ', method, ' ***')
    print('Min. score: ', min(scores))
    print('Max. score: ', max(scores))
    print('Avg. score: ', sum(scores) / len(scores))


toy = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1])
complete_data_space = np.array(list(itertools.product([0, 1], repeat=10)))
manhattan_scorer = ManhattanDistance(toy)

columns = ['method', 'min', 'max', 'avg', 'eps_psi', 'eps', 'n']
results = []

for eps_psi in [0.125, 0.25, 0.5, 1, 2, 4]:
    print('Exponential mechanism privacy budget set to ', eps_psi)

    exponential = ExponentialMechanism(manhattan_scorer, epsilon=eps_psi)

    # COMPLETE DATA SPACE

    scores = []
    for _ in range(TRIALS):
        out = exponential.privatize(complete_data_space)
        scores.append(manhattan_scorer.score_function(out))
    score_stats('COMPLETE DATA SPACE', scores)
    results.append(['complete', min(scores), max(scores), sum(scores) / len(scores), eps_psi, '', ''])




    # RANDOM SAMPLING
    for i, n in enumerate([1, 2, 4, 8, 16, 32, 64]):
        experiment_seed = i*1000
        scores = []
        for t in range(TRIALS):
            trial_seed = t*100
            sampler = RandomGenerator(base_seed=experiment_seed + trial_seed)
            output_space = [sampler.privatize_np(toy, relative_seed=i) for i in range(n)]
            out = exponential.privatize(output_space)
            scores.append(manhattan_scorer.score_function(out))
        score_stats('RANDOM SAMPLING WITH n=' + str(n), scores)
        results.append(['random', min(scores), max(scores), sum(scores) / len(scores), eps_psi, '', n])

    # EXTENDED RANDOMIZED RESPONSEg
    for j, eps in enumerate([0.125, 0.25, 0.5, 1, 1.4, 2, 4, 8]):
        randomized_seed = j * 5000

        for i, n in enumerate([1, 2, 4, 8, 16, 32, 64]):
            experiment_seed = i * 1000
            scores = []
            for t in range(TRIALS):
                trial_seed = t * 100
                sampler = RandomizeResponse(eps, base_seed=randomized_seed + experiment_seed + trial_seed)
                output_space = [sampler.privatize_np(toy, relative_seed=i) for i in range(n)]
                out = exponential.privatize(output_space)
                scores.append(manhattan_scorer.score_function(out))
            score_stats('EXTENDED RANDOMIZED RESPONSE WITH eps=' + str(eps) + ' and n=' + str(n), scores)
            results.append(['extended', min(scores), max(scores), sum(scores) / len(scores), eps_psi, eps, n])


data = pd.DataFrame(results, columns=columns)
data.to_csv('toy.csv', index=False)
