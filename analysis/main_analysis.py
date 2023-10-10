import pandas as pd
from src.loader.paths import metrics_path
from analysis.score_analysis import  compute_metrics

datasets = ['facebook_books']
data_types = ['clean']
# score_types = ['jaccard', 'euclidean', 'manhattan']
score_types = ['manhattan']
# epss = ['1.0']
epss = ['0.5', '1.0', '2.0', '3.0', '5.0', '10.0']
decimals = [3]
#generations = list(range(100, 100000, 100))
generations = [None]

result = []
for d in datasets:
    for d_t in data_types:
        for s_t in score_types:
            for e in epss:
                for g in generations:
                    metrics = []
                    for dec in decimals:
                        metrics += compute_metrics(dataset_name=d, data_type=d_t, score_type=s_t,
                                                  eps=e, decimal=dec, generations=g)
                    result.append(metrics)

        stats = pd.DataFrame(result, columns=['dataset', 'data_type', 'eps', 'transactions', 'size',
                                              'score_type', 'n_scores', 'min', 'max', 'mean', 'std'])
        stats_path = metrics_path(dataset_name=d, data_type=d_t)
        stats.to_csv(stats_path, sep='\t', index=False, decimal=',')
        print(f'Metrics stored at \'{stats_path}\' for {d} type {d_t}')