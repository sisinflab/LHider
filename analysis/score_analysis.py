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


def compute_metrics(dataset_name: str, data_type: str, score_type: str, eps: str, decimal: int, show=False, store=False):
    d_name = dataset_name
    d_type = data_type
    s_type = score_type
    eps = eps
    dec = decimal

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
                   generations=None)
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


def metrics_over_generations(dataset_name: str, data_type: str, score_type: str, eps: str,
                             gen_min: int, gen_max: int, gen_step: int):
    d_name = dataset_name
    d_type = data_type
    s_type = score_type
    eps = eps
    g_min = gen_min
    g_max = gen_max
    g_step = gen_step

    generations = list(range(g_min, g_max, g_step))

    # output directory
    o_dir = score_analysis_dir(dataset_name=d_name, dataset_type=d_type)

    create_directory(o_dir)

    score_loader = ScoreLoader(dataset_name=d_name,
                               dataset_type=d_type,
                               score_type=s_type,
                               eps=eps)
    score_values = score_loader.load()

    metrics = []
    for g in generations:
        scores = Score(score_values,
                       dataset_name=d_name,
                       dataset_type=d_type,
                       score_type=s_type,
                       eps=eps,
                       generations=g)
        # metrics = ['dataset', 'data_type', 'eps', 'score_type', 'generations', 'min', 'max', 'mean', 'std']
        metrics.append([d_name, d_type, eps, s_type, g, scores.min(), scores.max(), scores.mean(), scores.std()])

    stats = pd.DataFrame(metrics, columns=['dataset', 'data_type', 'eps', 'score_type', 'generations',
                                           'min', 'max', 'mean', 'std'])

    stats_output_path = metrics_over_generations_path(d_name, d_type, g_min, g_max, g_step)
    stats.to_csv(stats_output_path, sep='\t', index=False, decimal=',')
    print(f'Metrics file stored at \'{stats_output_path}\'')
    return stats


def over_generation_plot(metrics: list, dataset_name: str, data_type: str, score_type: str, eps: str,
                         gen_min: int, gen_max: int, gen_step: int, store: bool=True, show: bool=True):
    """
    Given a list of metrics it produces a plot that shows the values of the metrics over the generations.
    If the metrics are more than one, then the metrics are plotted in the same plot
    @param metrics: list of metrics
    @param dataset_name: name of the dataset
    @param data_type: type of dataset (clean or raw)
    @param score_type: type of score
    @param eps: epsilon value
    @param gen_min: number of generations from which the analysis begin
    @param gen_max: number of generations where the analysis ends
    @param gen_step: number of generation steps
    @param store: if true stores the image in a file
    @param store: if true shows the plot
    """
    # assert m in {'min', 'max', 'mean', 'std'}, f'Unknown metric \'{metric}\''
    ACCEPTED_METRICS = {'min', 'max', 'mean', 'std'}

    d_name = dataset_name
    d_type = data_type
    s_type = score_type
    eps = eps
    g_min = gen_min
    g_max = gen_max
    g_step = gen_step

    stats_output_path = metrics_over_generations_path(d_name, d_type, g_min, g_max, g_step)

    if os.path.exists(stats_output_path):
        stats = pd.read_csv(stats_output_path, sep='\t', decimal=',', header=0)
    else:
        stats = metrics_over_generations(d_name, d_type, s_type, eps, g_min, g_max, g_step)

    for m in metrics:
        assert m in ACCEPTED_METRICS, f'Unknown metric \'{m}\''
        values = stats['generations'].to_list(), stats[m].to_list()
        plt.plot(*values, label=m)


    if store:
        # output directory
        o_dir = score_analysis_dir(d_name, d_type)
        create_directory(o_dir)
        # output path
        metrics_name = '_'.join(metrics)
        o_path = os.path.join(o_dir, metrics_name + f'_over_generations_{g_min}_to_{g_max}_step_{g_step}.png')
        plt.savefig(o_path)
        print(f'Plot stored at \'{o_path}\'')

    if show:
        plt.show()


def over_generation_utility(dataset_name, data_type, score_type, eps,
                            threshold: float, gen_min: int, gen_max: int, gen_step: int):

    d_name = dataset_name
    d_type = data_type
    s_type = score_type
    eps = eps
    th = threshold
    g_min = gen_min
    g_max = gen_max
    g_step = gen_step

    stats_output_path = metrics_over_generations_path(d_name, d_type, g_min, g_max, g_step)

    score_loader = ScoreLoader(dataset_name=d_name,
                               dataset_type=d_type,
                               score_type=s_type,
                               eps=eps)
    score_values = score_loader.load()

    result = []
    generations = list(range(g_min, g_max, g_step))
    for g in generations:
        scores = Score(score_values,
                       dataset_name=d_name,
                       dataset_type=d_type,
                       score_type=s_type,
                       eps=eps,
                       generations=g)
        result.append(scores.values_over_threshold(thresh=th))

    stats = pd.DataFrame(zip(generations, result), columns=['generations', ''])

    plt.plot(generations, result)
    plt.show()

    stats_output_path = threshold_over_generations_path(d_name, d_type, g_min, g_max, g_step)
    stats.to_csv(stats_output_path, sep='\t', index=False, decimal=',')
    print(f'Values of threshold file stored at \'{stats_output_path}\'')
    return stats


over_generation_utility(dataset_name='facebook_books', data_type='clean', score_type='jaccard', eps='1.0',
                        threshold=0.0344, gen_min=1000, gen_max=100000, gen_step=1000)


# over_generation_plot(metrics=['min', 'mean', 'max'], dataset_name='facebook_books', data_type='clean', score_type='manhattan', eps='1.0',
#                      gen_min=100, gen_max=100000, gen_step=1000)

# compute_metrics(dataset_name='facebook_books', data_type='clean', score_type='jaccard', eps='1.0', decimal=1)
# da fare: vedere come varia il numero di valori oltre una certa soglia al crescere delle generazioni