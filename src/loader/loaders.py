from .paths import *
import pandas as pd
from scipy.sparse import csr_matrix
from src.loader.paths import *
import pickle


class TsvLoader:

    def __init__(self, path, return_type=None, directory='', main_directory='', header=None, names=None):
        assert isinstance(path, str), f'{self.__class__.__name__}: path must be a string. \n' \
                                      f'value: {path}\n' \
                                      f'type: {type(path)}'

        self.path = self.find_the_right_path(path, relative_directory=directory, main_directory=main_directory)

        if return_type is None:
            return_type = 'dataframe'

        return_types = {
            'dataframe': pd.DataFrame,
            'csr': csr_matrix
        }
        return_type = return_types[return_type]


        self._return_functions = {pd.DataFrame: self._load_dataframe,
                                  csr_matrix: self._load_crs}
        self.accepted_types = self._return_functions.keys()
        assert return_type in self.accepted_types, f'{self.__class__.__name__}: return type not managed by the loader.'
        self._return_type = return_type

        self.header = header
        self.names = names

    def load(self):
        data = pd.read_csv(self.path, sep='\t', header=self.header, names=self.names)
        return_function = self._return_functions[self._return_type]
        return return_function(data)

    def _load_dataframe(self, data):
        return data

    def _load_crs(self, data):
        return csr_matrix(data.pivot(index=0, columns=1, values=2).fillna(0))

    def find_the_right_path(self, path, relative_directory=None, main_directory=None):

        if relative_directory is None:
            relative_directory = ''
        if main_directory is None:
            main_directory = ''

        for p in [path, relative_directory, main_directory]:
            assert isinstance(p, str), f'must be a string. Found {p} with type {type(p)}'

        if main_directory:
            path_from_main = os.path.join(main_directory, relative_directory, path)
            assert os.path.exists(path_from_main), f'{self.__class__.__name__}: ' \
                                                   f'path \'{path_from_main}\' does not exists.'
            return path_from_main

        path_from_data_dir = os.path.join(DATA_DIR, relative_directory, path)
        assert os.path.exists(path_from_data_dir), f'{self.__class__.__name__}: ' \
                                                   f'path \'{path_from_data_dir}\' does not exists.'
        return path_from_data_dir


class ScoreLoader:
    def __init__(self, dataset_name, dataset_type, score_type, eps):
        self._dataset_name = dataset_name
        self._dataset_type = dataset_type
        self._score_type = score_type
        self._eps = eps

        self._score_dir = score_directory(dataset_name=dataset_name,
                                          dataset_type=dataset_type,
                                          score_type=score_type,
                                          eps_rr=eps)

    def load(self):
        score_path = scores_file_path(self._score_dir)
        with open(score_path, 'rb') as file:
            scores = pickle.load(file)
        return scores
