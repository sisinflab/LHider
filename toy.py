import numpy as np
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

for eps_psi in [0.125, 0.25, 0.5, 1, 2, 4]:
    print('Exponential mechanism privacy budget set to ', eps_psi)

    exponential = ExponentialMechanism(manhattan_scorer, epsilon=eps_psi)

    # COMPLETE DATA SPACE

    scores = []
    for _ in range(TRIALS):
        out = exponential.privatize(complete_data_space)
        scores.append(manhattan_scorer.score_function(out))
    score_stats('COMPLETE DATA SPACE', scores)

    # RANDOM SAMPLING
    sampler = RandomGenerator()
    output_space = [sampler.privatize_np(toy, relative_seed=i) for i in range(N)]

    for n in [1, 2, 4, 8, 16, 32, 64]:

        scores = []
        for _ in range(TRIALS):
            out = exponential.privatize(output_space[:n])
            scores.append(manhattan_scorer.score_function(out))
        score_stats('RANDOM SAMPLING WITH n=' + str(n), scores)

    # EXTENDED RANDOMIZED RESPONSE
    for eps in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
        sampler = RandomizeResponse(eps)
        output_space = [sampler.privatize_np(toy, relative_seed=i) for i in range(N)]


        for n in [1, 2, 4, 8, 16, 32, 64]:

            scores = []
            for _ in range(TRIALS):
                out = exponential.privatize(output_space[:n])
                scores.append(manhattan_scorer.score_function(out))
            score_stats('RANDOM SAMPLING WITH eps=' + str(eps) + ' and n=' + str(n), scores)


print()