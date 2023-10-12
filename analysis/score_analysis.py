import math
import os

import pandas as pd
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


class ScoreAnalyzer:
    def __init__(self, dataset_name: str, data_type: str, score_type: str, eps: str):
        self._dataset_name = dataset_name
        self._data_type = data_type
        self._score_type = score_type
        self._eps = eps
        self._score_values = self.load_score_values()
        self._scores = self.score_object(score_values=self._score_values)
        self._output_dir = score_analysis_dir(dataset_name=dataset_name, dataset_type=data_type)
        create_directory(self._output_dir)

        self._score_distribution_output_dir = os.path.join(self._output_dir, 'score_distribution')
        self._metrics_over_generation_output_dir = os.path.join(self._output_dir, 'over_generations')

        self.dataset_path = dataset_filepath(self._dataset_name, self._data_type)
        loader = TsvLoader(self.dataset_path)
        self.dataset = DPDataFrame(loader.load())

    def load_score_values(self, dataset_name: str = None, dataset_type: str = None, score_type: str = None, eps: str = None):

        if dataset_name is None:
            dataset_name = self._dataset_name
        if dataset_type is None:
            dataset_type = self._data_type
        if score_type is None:
            score_type = self._score_type
        if eps is None:
            eps = self._eps

        score_loader = ScoreLoader(dataset_name=dataset_name,
                                   dataset_type=dataset_type,
                                   score_type=score_type,
                                   eps=eps)
        return score_loader.load()

    def score_object(self, score_values: dict = None, dataset_name: str = None, dataset_type: str = None, score_type: str = None,
                          eps: str = None, generations: int = None):

        if dataset_name is None:
            dataset_name = self._dataset_name
        if dataset_type is None:
            dataset_type = self._data_type
        if score_type is None:
            score_type = self._score_type
        if eps is None:
            eps = self._eps

        if score_values is None:
            score_values = self.load_score_values(dataset_name, dataset_type, score_type, eps)

        scores = Score(scores=score_values,
                       dataset_name=self._dataset_name,
                       dataset_type=self._data_type,
                       score_type=self._score_type,
                       eps=self._eps,
                       generations=generations)

        return scores

    def discrete_scores(self, decimal):
        return np.round(self._scores.data, decimal)

    def count_scores(self, scores):
        if scores is None:
            scores = self._score_values
        count_scores = Counter(scores)
        ordered_scores = sorted(scores)
        count_ordered_scores = {k: count_scores[k] for k in ordered_scores}
        return list(count_ordered_scores.keys()), list(count_ordered_scores.values())

    def store_distribution_metrics(self, metrics):

        o_dir = self._score_distribution_output_dir
        assert os.path.exists(o_dir), f'Missing directory at \'{o_dir}\''

        output_file = os.path.join(o_dir, 'distribution_metrics.tsv')
        stats = pd.DataFrame([metrics], columns=['dataset_name', 'data_type', 'eps', 'transactions', 'size', 'score_type',
                                               'generations', 'min', 'max', 'mean', 'std'])

        if os.path.exists(output_file):
            stored_data = pd.read_csv(output_file, sep='\t', header=0, decimal=',')
            if not (np.array(stats) == stored_data.values).all(1).any():
                pd.concat([stored_data, stats]).to_csv(output_file, sep='\t', index=False, decimal=',')
                print(f'Updated stats stored at \'{output_file}\'')
        else:
            stats.to_csv(output_file, sep='\t', index=False, decimal=',')
            print(f'Stats stored at \'{output_file}\'')

    def score_distribution(self, decimal, plot=True, store_plot=False, store_stats=False):

        discrete_scores = self.discrete_scores(decimal=decimal)
        x, y = self.count_scores(discrete_scores)

        o_dir = self._score_distribution_output_dir
        if store_stats or store_plot:
            create_directory(o_dir)

        if plot:
            plt.clf()
            plt.plot(x, y)
            plt.show()
            plt.clf()

        if store_plot:
            if not plot:
                plt.clf()
                plt.plot(x, y)

            output_file = os.path.join(o_dir, self._score_type + f'_{decimal}' + '.png')
            plt.savefig(output_file)
            print(f'Plot stored at \'{output_file}\'')
            plt.clf()

        if store_stats:
            metrics = [self._dataset_name, self._data_type, self._eps,
                       self.dataset.transactions, self.dataset.size, self._score_type,
                       len(self._scores), self._scores.min(), self._scores.max(), self._scores.mean(), self._scores.std()]
            self.store_distribution_metrics(metrics)

    def metrics_generation_path(self, gen_min, gen_max, gen_step):
        o_dir = self._metrics_over_generation_output_dir
        create_directory(o_dir)
        return os.path.join(o_dir, f'{self._score_type}_{self._eps}_{gen_min}->{gen_max}-{gen_step}.tsv')


    def default_generation_values(self, gen_min, gen_max, gen_step):

        g_min = gen_min
        g_max = gen_max
        g_step = gen_step

        if gen_min is None:
            g_min = 0
        if gen_max is None:
            g_max = len(self._scores)
        if gen_step is None:
            g_step = round((g_max-g_min)/100)

        return g_min, g_max, g_step

    def metrics_over_generation(self, gen_min: int = None, gen_max: int = None, gen_step: int = None, store_stats=False):

        g_min, g_max, g_step = self.default_generation_values(gen_min, gen_max, gen_step)

        generations = list(range(g_min, g_max, g_step))
        o_dir = self._metrics_over_generation_output_dir
        create_directory(o_dir)

        metrics = []
        for g in generations:
            scores = self.score_object(self._score_values, generations=g)
            metrics.append(
                [self._dataset_name, self._data_type, self._eps,
                 self.dataset.transactions, self.dataset.size, self._score_type,
                 len(scores), scores.min(), scores.max(), scores.mean(), scores.std()])

        stats = pd.DataFrame(metrics, columns=['dataset_name', 'data_type', 'eps', 'transactions', 'size', 'score_type', 'generations', 'min', 'max', 'mean', 'std'])

        # output file
        if store_stats:
            o_file = self.metrics_generation_path(g_min, g_max, g_step)
            stats.to_csv(o_file, sep='\t', index=False, decimal=',')
            print(f'Metrics file stored at \'{o_file}\'')
        else:
            print(stats)

        return stats

    def plot_metrics_over_generation(self, metrics:list, gen_min: int = None, gen_max: int = None, gen_step: int = None, show=False, store=False):
        ACCEPTED_METRICS = {'min', 'max', 'mean', 'std'}

        g_min, g_max, g_step = self.default_generation_values(gen_min, gen_max, gen_step)

        stats_path = self.metrics_generation_path(g_min, g_max, g_step)

        if os.path.exists(stats_path):
            stats = pd.read_csv(stats_path, sep='\t', decimal=',', header=0)
        else:
            stats = self.metrics_over_generation(g_min, g_max, g_step)

        for m in metrics:
            assert m in ACCEPTED_METRICS, f'Unknown metric \'{m}\''
            values = stats['generations'].to_list(), stats[m].to_list()
            plt.plot(*values, label=m)

        if store:
            # output directory
            o_dir = self._metrics_over_generation_output_dir
            # output file
            o_file = os.path.join(o_dir, f'plot_{self._score_type}_{self._eps}_{g_min}->{g_max}-{g_step}.png')
            plt.savefig(o_file)
            print(f'Image stored at \'{o_file}\'')

        if show:
            plt.show()
        plt.clf()


# def score_distribution(dataset_name: str, data_type: str, score_type: str, eps: str, decimal: int, show=False,
#                        store=False):
#     d_name = dataset_name
#     d_type = data_type
#     s_type = score_type
#     eps = eps
#     dec = decimal
#
#     # output directory
#     o_dir = score_analysis_dir(dataset_name=d_name, dataset_type=d_type)
#
#     create_directory(o_dir)
#
#     d_path = dataset_filepath(d_name, d_type)
#     loader = TsvLoader(d_path)
#     data = loader.load()
#     dataset = DPDataFrame(data)
#     transactions = dataset.transactions
#     size = dataset.size
#
#     score_loader = ScoreLoader(dataset_name=d_name,
#                                dataset_type=d_type,
#                                score_type=s_type,
#                                eps=eps)
#     score_values = score_loader.load()
#
#     scores = Score(score_values,
#                    dataset_name=d_name,
#                    dataset_type=d_type,
#                    score_type=s_type,
#                    eps=eps,
#                    generations=None)
#     # metrics = ['dataset', 'data_type', 'eps', 'transactions', 'size',
#     # 'score_type', 'n_scores', 'min', 'max', 'mean', 'std']
#     metrics = [d_name, d_type, eps, transactions, size,
#                s_type, len(scores), scores.min(), scores.max(), scores.mean(), scores.std()]
#     discrete_scores = score_discretize_and_count(scores, decimal=dec)
#
#     if show:
#         show_plot(discrete_scores)
#     if store:
#         store_plot(scores, discrete_scores, decimal, o_dir)
#
#     plt.clf()
#     return metrics
#
#
# def metrics_over_generations(dataset_name: str, data_type: str, score_type: str, eps: str,
#                              gen_min: int, gen_max: int, gen_step: int):
#     d_name = dataset_name
#     d_type = data_type
#     s_type = score_type
#     eps = eps
#     g_min = gen_min
#     g_max = gen_max
#     g_step = gen_step
#
#     generations = list(range(g_min, g_max, g_step))
#
#     # output directory
#     o_dir = score_analysis_dir(dataset_name=d_name, dataset_type=d_type)
#
#     create_directory(o_dir)
#
#     score_loader = ScoreLoader(dataset_name=d_name,
#                                dataset_type=d_type,
#                                score_type=s_type,
#                                eps=eps)
#     score_values = score_loader.load()
#
#     metrics = []
#     for g in generations:
#         scores = Score(score_values,
#                        dataset_name=d_name,
#                        dataset_type=d_type,
#                        score_type=s_type,
#                        eps=eps,
#                        generations=g)
#         # metrics = ['dataset', 'data_type', 'eps', 'score_type', 'generations', 'min', 'max', 'mean', 'std']
#         metrics.append([d_name, d_type, eps, s_type, g, scores.min(), scores.max(), scores.mean(), scores.std()])
#
#     stats = pd.DataFrame(metrics, columns=['dataset', 'data_type', 'eps', 'score_type', 'generations',
#                                            'min', 'max', 'mean', 'std'])
#
#     stats_output_path = metrics_over_generations_path(d_name, d_type, g_min, g_max, g_step)
#     stats.to_csv(stats_output_path, sep='\t', index=False, decimal=',')
#     print(f'Metrics file stored at \'{stats_output_path}\'')
#     return stats
#
#
# def over_generation_plot(metrics: list, dataset_name: str, data_type: str, score_type: str, eps: str,
#                          gen_min: int, gen_max: int, gen_step: int, store: bool = True, show: bool = True):
#     """
#     Given a list of metrics it produces a plot that shows the values of the metrics over the generations.
#     If the metrics are more than one, then the metrics are plotted in the same plot
#     @param metrics: list of metrics
#     @param dataset_name: name of the dataset
#     @param data_type: type of dataset (clean or raw)
#     @param score_type: type of score
#     @param eps: epsilon value
#     @param gen_min: number of generations from which the analysis begin
#     @param gen_max: number of generations where the analysis ends
#     @param gen_step: number of generation steps
#     @param store: if true stores the image in a file
#     @param store: if true shows the plot
#     """
#     # assert m in {'min', 'max', 'mean', 'std'}, f'Unknown metric \'{metric}\''
#     ACCEPTED_METRICS = {'min', 'max', 'mean', 'std'}
#
#     d_name = dataset_name
#     d_type = data_type
#     s_type = score_type
#     eps = eps
#     g_min = gen_min
#     g_max = gen_max
#     g_step = gen_step
#
#     stats_output_path = metrics_over_generations_path(d_name, d_type, g_min, g_max, g_step)
#
#     if os.path.exists(stats_output_path):
#         stats = pd.read_csv(stats_output_path, sep='\t', decimal=',', header=0)
#     else:
#         stats = metrics_over_generations(d_name, d_type, s_type, eps, g_min, g_max, g_step)
#
#     for m in metrics:
#         assert m in ACCEPTED_METRICS, f'Unknown metric \'{m}\''
#         values = stats['generations'].to_list(), stats[m].to_list()
#         plt.plot(*values, label=m)
#
#     if store:
#         # output directory
#         o_dir = score_analysis_dir(d_name, d_type)
#         create_directory(o_dir)
#         # output path
#         metrics_name = '_'.join(metrics)
#         o_path = os.path.join(o_dir, metrics_name + f'_over_generations_{g_min}_to_{g_max}_step_{g_step}.png')
#         plt.savefig(o_path)
#         print(f'Plot stored at \'{o_path}\'')
#
#     if show:
#         plt.show()


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


def compare_metric_with_manhattan(dataset_name, data_type, score_type: list, eps, generations: int = None):
    d_name = dataset_name
    d_type = data_type
    s_type = score_type
    eps = eps
    g = generations

    score_loader = ScoreLoader(dataset_name=d_name,
                               dataset_type=d_type,
                               score_type='manhattan',
                               eps=eps)
    manhattan_score_values = score_loader.load()

    score_loader = ScoreLoader(dataset_name=d_name,
                               dataset_type=d_type,
                               score_type=s_type,
                               eps=eps)
    score_values = score_loader.load()

    # associate to each score the corresponding manhattan distance score
    score_and_man = [(s, manhattan_score_values[k]) for k, s in score_values.items() if k in manhattan_score_values]
    score_and_man = sorted(score_and_man, key=lambda x: x[0])

    print(f'{len(score_and_man)} scores have been paired with the corresponding manhattan distance')

    log_score_and_man = [(math.log(s), math.log(m)) for s, m in score_and_man]

    x, y = list(zip(*score_and_man))
    log_x, log_y = list(zip(*log_score_and_man))

    plt.plot(x, y)
    plt.show()
    plt.clf()

    plt.plot(log_x, log_y)
    plt.show()
    plt.clf()

    print()


# compare_metric_with_manhattan(dataset_name='facebook_books', data_type='clean', score_type='jaccard', eps='1.0')

# over_generation_utility(dataset_name='facebook_books', data_type='clean', score_type='jaccard', eps='1.0',
#                         threshold=0.0344, gen_min=1000, gen_max=100000, gen_step=1000)


# over_generation_plot(metrics=['min', 'mean', 'max'], dataset_name='facebook_books', data_type='clean', score_type='manhattan', eps='1.0',
#                      gen_min=100, gen_max=100000, gen_step=1000)

# compute_metrics(dataset_name='facebook_books', data_type='clean', score_type='jaccard', eps='1.0', decimal=1)
# da fare: vedere come varia il numero di valori oltre una certa soglia al crescere delle generazioni

analyzer = ScoreAnalyzer(dataset_name='facebook_books', data_type='clean', score_type='jaccard', eps='2.0')
analyzer.score_distribution(decimal=4, plot=True, store_plot=True, store_stats=True)
analyzer.metrics_over_generation()
analyzer.plot_metrics_over_generation(metrics=['std'], store=True, show=True)
