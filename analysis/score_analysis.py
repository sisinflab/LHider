from matplotlib import pyplot as plt
from collections import Counter
from src.loader.loaders import *
from src.exponential_mechanism.scores import *
from src.loader.paths import create_directory
from src.dataset.dataset import DPDataFrame


def score_to_dataframe(scores: list, decimal=4):
    scores = [round(s, decimal) for s in scores]
    count_scores = Counter(scores)
    ordered_scores = sorted(scores)
    count_ordered_scores = {k: count_scores[k] for k in ordered_scores}
    return pd.DataFrame(data=count_ordered_scores.items(), columns=['score', 'freq'])


def score_discretize_and_count(score: Score, decimal=4):
    rounded_scores = np.round(score.data, decimal)
    count_scores = Counter(rounded_scores)
    ordered_scores = sorted(rounded_scores)
    count_ordered_scores = {k: count_scores[k] for k in ordered_scores}
    return list(count_ordered_scores.keys()), list(count_ordered_scores.values())


def show_plot(discrete_scores):
    plt.plot(*discrete_scores)
    plt.show()


def store_plot(scores: Score, discrete_scores: tuple,
               decimal: float, output_dir: str, debug=True):
    plt.plot(*discrete_scores)
    output_file = os.path.join(output_dir, scores.score_name(decimal) + '.png')
    plt.savefig(output_file)
    if debug:
        print(f'Plot stored at \'{output_file}\'')


def score_plot(score: Score, decimal=4):
    rounded_scores = np.round(score.data, decimal)
    count_scores = Counter(rounded_scores)
    ordered_scores = sorted(rounded_scores)
    count_ordered_scores = {k: count_scores[k] for k in ordered_scores}
    x = plt.plot(list(count_ordered_scores.keys()),
                 list(count_ordered_scores.values()))
    plt.show()
    return x


def run(dataset_name: str, data_type: str, score_type: str, eps: str, decimal: int,
        generations: int = None, show=False, store=False):
    d_name = dataset_name
    d_type = data_type
    s_type = score_type
    eps = eps
    dec = decimal
    gen = generations

    # output directory
    o_dir = score_analysis_dir(dataset_name=d_name, dataset_type=d_type)

    create_directory(o_dir)

    d_path = dataset_filepath(d_name, d_type)
    loader = TsvLoader(d_path)
    data = loader.load()
    dataset = DPDataFrame(data)
    transactions = dataset.transactions
    size = dataset.size

    score_loader = ScoreLoader(dataset_name=d_name,
                               dataset_type=d_type,
                               score_type=s_type,
                               eps=eps)
    score_values = score_loader.load()

    scores = Score(score_values,
                   dataset_name=d_name,
                   dataset_type=d_type,
                   score_type=s_type,
                   eps=eps,
                   generations=gen)
    # metrics = ['dataset', 'data_type', 'eps', 'transactions', 'size',
    # 'score_type', 'n_scores', 'min', 'max', 'mean', 'std']
    metrics = [d_name, d_type, eps, transactions, size,
               s_type, len(scores), scores.min(), scores.max(), scores.mean(), scores.std()]
    discrete_scores = score_discretize_and_count(scores, decimal=dec)

    if show:
        show_plot(discrete_scores)
    if store:
        store_plot(scores, discrete_scores, decimal, o_dir)

    plt.clf()
    return metrics


datasets = ['facebook_books']
data_types = ['clean']
score_types = ['jaccard', 'euclidean']
epss = ['1.0']
decimals = [3]
generations = list(range(100, 100000, 100))

result = []
for d in datasets:
    for d_t in data_types:
        for s_t in score_types:
            for e in epss:
                for g in generations:
                    metrics = []
                    for dec in decimals:
                        metrics = run(dataset_name=d, data_type=d_t, score_type=s_t, eps=e, decimal=dec, generations=g)
                    result.append(metrics)

        stats = pd.DataFrame(result, columns=['dataset', 'data_type', 'eps', 'transactions', 'size',
                                              'score_type', 'n_scores', 'min', 'max', 'mean', 'std'])
        stats_path = metrics_path(dataset_name=d, data_type=d_t)
        stats.to_csv(stats_path, sep='\t')
        print(f'Metrics stored at \'{stats_path}\' for {d} type {d_t}')
