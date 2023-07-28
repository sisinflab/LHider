import pandas as pd
import os


def load_kg(path, header=0):
    return pd.read_csv(path, sep='\t', header=header, names=['s', 'p', 'o'])


def load_dataset(path):
    return pd.read_csv(path, sep='\t', header=None, names=['u', 'i', 'r'])


def store_dataset(data, folder=None, name=None, message=None, **kwargs):
    if folder is None:
        folder = '.'
    if name is None:
        name = 'dataset'
    if message is None:
        message = 'dataset'

    if os.path.exists(folder) is False:
        os.makedirs(folder)

    dataset_path = os.path.abspath(os.path.join(folder, name)) + '.tsv'
    data.to_csv(dataset_path, sep='\t', header=None, index=None)
    print(f'{message.capitalize()} stored at \'{dataset_path}\'')
    return {name: dataset_path}


class Filter:
    def __init__(self, **kwargs):
        self._flag = False
        self._output = dict()

    def filter_engine(self):
        pass

    def filter_output(self):
        return self._output

    @property
    def flag(self):
        return self._flag

    def filter(self):
        self.filter_engine()
        return self.filter_output()


class FilterPipeline(Filter):
    def __init__(self, filters, **kwargs):
        super(FilterPipeline, self).__init__()
        self._filters = filters
        self._kwargs = kwargs

    def filter_engine(self):
        for f in self._filters:
            self._kwargs.update(f(**self._kwargs).filter())

    def filter_output(self):
        return self._kwargs
