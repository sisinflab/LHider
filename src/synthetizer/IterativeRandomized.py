from src.dataset.dataset import DPDataset
from src.loader.loaders import TsvLoader


class IterativeRandomized:

    def __init__(self, dataset, switch_probability):

        # source dataset to be anonymized
        self._d = dataset
        self._p = switch_probability
        self._q = 1 - self._p

    def generate(self):
        self._d.to_numpy()

    def output_probability(self, y):
        pass


dataset_path = '/home/alberto/PycharmProjects/ExponentialMechanismForRecommenderSystems/data/YahooMovies.tsv'
loader = TsvLoader(path=dataset_path, return_type='dataframe')
data = DPDataset(loader.load(), path=dataset_path, name='yahoo', columns=['u', 'i', 'r'])
randomizer = IterativeRandomized(dataset=data, switch_probability=0.5)
randomizer.generate()
