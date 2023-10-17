import math

import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
from src.loader.loaders import *
from src.exponential_mechanism.scores import *
from src.loader.paths import create_directory
from src.dataset.dataset import DPDataFrame


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
        self._utility_output_dir = os.path.join(self._output_dir, 'utility_over_generations')
        self._compare_metric_with_manhattan_output_dir = os.path.join(self._output_dir, 'metric_with_manhattan')

        self.dataset_path = dataset_filepath(self._dataset_name, self._data_type)
        loader = TsvLoader(self.dataset_path)
        self.dataset = DPDataFrame(loader.load())

        self._accepted_metrics = {'min', 'max', 'mean', 'std'}

    def load_score_values(self, dataset_name: str = None, dataset_type: str = None, score_type: str = None,
                          eps: str = None) -> dict:

        dataset_name = (dataset_name or self._dataset_name)
        dataset_type = (dataset_type or self._data_type)
        score_type = (score_type or self._score_type)
        eps = (eps or self._eps)

        score_loader = ScoreLoader(dataset_name=dataset_name,
                                   dataset_type=dataset_type,
                                   score_type=score_type,
                                   eps=eps)
        return score_loader.load()

    def score_object(self, score_values: dict = None, dataset_name: str = None, dataset_type: str = None,
                     score_type: str = None, eps: str = None, generations: int = None) -> Score:

        dataset_name = (dataset_name or self._dataset_name)
        dataset_type = (dataset_type or self._data_type)
        score_type = (score_type or self._score_type)
        eps = (eps or self._eps)

        if score_values is None:
            score_values = self.load_score_values(dataset_name, dataset_type, score_type, eps)

        scores = Score(scores=score_values,
                       dataset_name=self._dataset_name,
                       dataset_type=self._data_type,
                       score_type=self._score_type,
                       eps=self._eps,
                       generations=generations)

        return scores

    def discrete_scores(self, decimal: int):
        return np.round(self._scores.data, decimal)

    def count_scores(self, scores) -> (list, list):
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
        stats = pd.DataFrame([metrics], columns=['dataset_name', 'data_type', 'eps', 'transactions', 'size',
                                                 'score_type', 'generations', 'min', 'max', 'mean', 'std'])

        if os.path.exists(output_file):
            stored_data = pd.read_csv(output_file, sep='\t', header=0, decimal=',')
            if not (np.array(stats) == stored_data.values).all(1).any():
                pd.concat([stored_data, stats]).to_csv(output_file, sep='\t', index=False, decimal=',')
                print(f'Updated stats stored at \'{output_file}\'')
        else:
            stats.to_csv(output_file, sep='\t', index=False, decimal=',')
            print(f'Stats stored at \'{output_file}\'')

    def score_distribution(self, decimal: int, show: bool = True, store_plot: bool = False,
                           store_stats: bool = False) -> None:

        discrete_scores = self.discrete_scores(decimal=decimal)
        x, y = self.count_scores(discrete_scores)

        o_dir = self._score_distribution_output_dir
        if store_stats or store_plot:
            create_directory(o_dir)

        if show or store_plot:
            plt.clf()
            plt.plot(x, y)
            plt.title("Score Distribution")
            plt.xlabel("scores")
            plt.ylabel("number of datasets")

        if store_plot:
            output_file = os.path.join(o_dir, f'{self._score_type}_eps_{self._eps}_dec_{decimal}.png')
            plt.savefig(output_file)
            print(f'Plot stored at \'{output_file}\'')
            plt.clf()
            output_file = os.path.join(o_dir, f'{self._score_type}_eps_{self._eps}_dec_{decimal}.tsv')
            df = pd.DataFrame(zip(x, y), columns=["score", "freq"]).sort_values(["score"])
            print(f'Score frequency stored at \'{output_file}\'')
            df.to_csv(output_file, sep="\t")

        if show:
            plt.show()
            plt.clf()

        if store_stats:
            metrics = [self._dataset_name, self._data_type, self._eps,
                       self.dataset.transactions, self.dataset.size, self._score_type,
                       len(self._scores), self._scores.min(), self._scores.max(),
                       self._scores.mean(), self._scores.std()]
            self.store_distribution_metrics(metrics)

    def metrics_generation_path(self, gen_min: int, gen_max: int, gen_step: int) -> str:

        o_dir = self._metrics_over_generation_output_dir
        create_directory(o_dir)

        return os.path.join(o_dir, f'generations_{gen_min}_to_{gen_max}-{gen_step}.tsv')

    def default_generation_values(self, gen_min: int, gen_max: int, gen_step: int) -> (int, int, int):

        g_min = (gen_min or 0)
        g_max = (gen_max or len(self._scores))
        g_step = (gen_step or round((g_max-g_min)/100))

        # if gen_min is None:
        #     g_min = 0
        # if gen_max is None:
        #     g_max = len(self._scores)
        # if gen_step is None:
        #     g_step = round((g_max-g_min)/100)

        return g_min, g_max, g_step

    def metrics_over_generation(self, gen_min: int = None, gen_max: int = None, gen_step: int = None,
                                store_stats: bool = False) -> pd.DataFrame:

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

        stats = pd.DataFrame(metrics, columns=['dataset_name', 'data_type', 'eps', 'transactions', 'size',
                                               'score_type', 'generations', 'min', 'max', 'mean', 'std'])

        # output file
        if store_stats:
            o_file = self.metrics_generation_path(g_min, g_max, g_step)
            stats.to_csv(o_file, sep='\t', index=False, decimal=',')
            print(f'Metrics file stored at \'{o_file}\'')
        else:
            print(stats)

        return stats

    def plot_metrics_over_generation(self, metrics: list, gen_min: int = None, gen_max: int = None,
                                     gen_step: int = None, show: bool = False, store_plot: bool = False) -> None:

        g_min, g_max, g_step = self.default_generation_values(gen_min, gen_max, gen_step)

        stats_path = self.metrics_generation_path(g_min, g_max, g_step)

        if os.path.exists(stats_path):
            stats = pd.read_csv(stats_path, sep='\t', decimal=',', header=0)
        else:
            stats = self.metrics_over_generation(g_min, g_max, g_step)

        for m in metrics:
            assert m in self._accepted_metrics, f'Unknown metric \'{m}\''
            values = stats['generations'].to_list(), stats[m].to_list()
            plt.plot(*values, label=m)

        plt.title(f"{metrics[0] if len(metrics) == 1 else 'Metrics'} Over Generations")
        plt.xlabel("number of generations")
        plt.ylabel("metrics")

        if len(metrics) > 1:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
            plt.tight_layout()

        if store_plot:
            # output directory
            o_dir = self._metrics_over_generation_output_dir
            # output file
            o_file = os.path.join(o_dir,
                                  f'{self._score_type}_eps_{self._eps}_{"_".join(metrics)}_{g_min}_to_{g_max}-{g_step}.png')
            plt.savefig(o_file)
            print(f'Image stored at \'{o_file}\'')

        if show:
            plt.show()
        plt.clf()

    def default_threshold(self) -> float:
        return self._scores.max() - (self._scores.max() - self._scores.min()) * 0.15

    def over_generation_utility_path(self, threshold: float) -> str:
        o_dir = self._utility_output_dir
        create_directory(o_dir)
        return os.path.join(o_dir, f'{self._score_type}_eps_{self._eps}_th_{threshold}.tsv')

    def over_generation_utility(self, threshold: float = None, gen_min: int = None, gen_max: int = None,
                                gen_step: int = None, show: bool = False, store_plot: bool = False,
                                store_stats: bool = True) -> pd.DataFrame:

        g_min, g_max, g_step = self.default_generation_values(gen_min, gen_max, gen_step)

        if threshold is None:
            threshold = self.default_threshold()
        th = threshold

        o_dir = self._utility_output_dir
        create_directory(o_dir)

        result = []
        generations = list(range(g_min, g_max, g_step))
        for g in generations:
            scores = self.score_object(self._score_values, generations=g)
            result.append(scores.values_over_threshold(thresh=th))
        stats = pd.DataFrame(zip(generations, result), columns=['generations', 'optimal_scores'])

        if store_plot or show:
            plt.xlabel("number of generations")
            plt.ylabel("number of score over threshold")
            plt.plot(generations, result)

        if store_plot:
            plot_file = os.path.join(self._utility_output_dir, f'{self._score_type}_eps_{self._eps}_th_{threshold}.png')
            plt.savefig(plot_file)
            print(f'Plot stored at \'{plot_file}\'')

        if show:
            plt.show()

        if store_stats:
            o_file = self.over_generation_utility_path(threshold=th)
            stats.to_csv(o_file, sep='\t', index=False, decimal=',')
            print(f'Values of threshold file stored at \'{o_file}\'')

        plt.clf()
        return stats

    def store_metrics_corr(self, data: pd.DataFrame, file_name) -> None:

        o_dir = self._compare_metric_with_manhattan_output_dir
        assert os.path.exists(o_dir), f'Missing directory at \'{o_dir}\''

        output_file = os.path.join(o_dir, f'{file_name}.tsv')

        if os.path.exists(output_file):
            stored_data = pd.read_csv(output_file, sep='\t', header=0, decimal=',')
            if not stored_data.equals(data):
                pd.concat([stored_data, data]).to_csv(output_file, sep='\t', index=False, decimal=',')
                print(f'Updated stats stored at \'{output_file}\'')
        else:
            data.to_csv(output_file, sep='\t', index=False, decimal=',')
            print(f'Stats stored at \'{output_file}\'')

    def compare_metric_with_manhattan(self, log: bool = False, show: bool = False,
                                      store_plot: bool = False, store_stats: bool = True) -> None:

        manhattan_score_values = self.load_score_values(score_type='manhattan')
        score_values = self.load_score_values(score_type=self._score_type)

        # associate to each score the corresponding manhattan distance score
        score_and_man = [(math.log(s), math.log(manhattan_score_values[k])) if log else (s, manhattan_score_values[k])
                         for k, s in score_values.items() if k in manhattan_score_values]
        score_and_man = sorted(score_and_man, key=lambda x: x[0])
        x, y = list(zip(*score_and_man))

        if store_stats or store_plot:
            o_dir = self._compare_metric_with_manhattan_output_dir
            create_directory(o_dir)

        if store_plot or show:
            plt.title("Scores Comparison")
            plt.xlabel(f"{self._score_type} score {'(log)' if log else ''}")
            plt.ylabel(f"manhattan score {'(log)' if log else ''}")
            plt.plot(x, y)
            plt.tight_layout()

        if store_plot:
            plot_file = os.path.join(self._compare_metric_with_manhattan_output_dir,
                                     f'{self._score_type}_eps_{self._eps}.png')
            plt.savefig(plot_file)
            print(f'Plot stored at \'{plot_file}\'')

        if show:
            plt.show()

        if store_stats:
            corr = np.corrcoef(x, y)[0][1]
            data = [self._dataset_name, self._data_type, self._eps, self._score_type, corr]
            stats = pd.DataFrame([data],
                                 columns=['dataset_name', 'data_type', 'eps', 'score_type', 'correlation'])
            self.store_metrics_corr(data=stats, file_name='correlation_manhattan')

        plt.clf()

        print(f'{len(score_and_man)} scores have been paired with the corresponding manhattan distance')


