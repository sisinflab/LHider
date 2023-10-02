import numpy as np
import pandas as pd
import os
import math
from scipy.sparse import csr_matrix


class Dataset:

    def __init__(self, path=None, data_name=None, writer=None, columns=None):

        # TODO: loader, writer, binarizer

        self._writer = writer

        # list of user
        self._users = None
        # count of different users
        self._n_users = None
        # list of itmes
        self._items = None
        # count of different items
        self._n_items = None
        # list of ratings
        self._ratings = None
        # count of different ratings
        self._n_ratings = None
        # count of transactions
        self._transactions = None

        self.dataset = None

        if not data_name:
            if path:
                data_name = os.path.split(path)[-1].split('.')[0]
            else:
                data_name = 'dataset'
        self._name = data_name

        self._columns = columns
        self._n_columns = None

        self._user_col = None
        self._item_col = None
        self._ratings_col = None
        self._timestamp_col = None

        self._user_public_to_private = None
        self._item_public_to_private = None
        self._user_private_to_public = None
        self._item_private_to_public = None

        self._binary = False

        # metrics
        self._size = None
        self._space_size_log = None
        self._shape_log = None
        self._density = None
        self._density_log = None
        self._gini_item = None
        self._gini_user = None
        self._metrics = {'space size log': self._space_size_log,
                         'shape log': self._shape_log,
                         'density': self._density,
                         'density log': self._density_log,
                         'gini item': self._gini_item,
                         'gini user': self._gini_user}

        # TODO: valutare se servono per davvero
        self._sorted_items = None
        self._sorted_users = None

    @property
    def values(self):
        return self.dataset.values

    def set_relevant_columns(self):

        assert self.columns is not None, f'{self.__class__.__name__}: assign columns before calling set_relevant_columns'

        # columns < 2
        if self.n_columns < 2:
            raise KeyError('dataset must have at least two columns: user and item')

        # columns >= 2
        self._user_col = self.columns[0]
        print(f'{self.__class__.__name__}: column 0 set as user column')
        self._item_col = self.columns[1]
        print(f'{self.__class__.__name__}: column 1 set as item column')

        if self.n_columns > 2:
            self._ratings_col = self.columns[2]
            print(f'{self.__class__.__name__}: column 2 set as ratings column')

        if self.n_columns > 3:
            self._timestamp_col = self.columns[3]
            print(f'{self.__class__.__name__}: column 3 set as timestamp column')

    @property
    def n_columns(self):
        assert self.columns is not None, f'{self.__class__.__name__}: columns must be assigned before calling n_columns'
        if self._n_columns is None:
            self._n_columns = len(self.columns)
        return self._n_columns

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, value):
        assert isinstance(value, list), f'{self.__class__.__name__}: columns must be assigned with a list'
        self._columns = value

    @property
    def user_col(self):

        assert self.columns is not None, f'{self.__class__.__name__}: columns must be assigned before calling user col'

        if self._user_col is None:
            if self.n_columns < 2:
                raise KeyError('dataset must have at least two columns for auto-setting user column')

            self._user_col = self.columns[0]
            print(f'{self.__class__.__name__}: first column set as user column')
        return self._user_col

    @property
    def item_col(self):
        assert self.columns is not None, f'{self.__class__.__name__}: columns must be assigned before calling item col'

        if self._item_col is None:
            if self.n_columns < 2:
                raise KeyError('dataset must have at least two columns for auto-setting item column')

            self._item_col = self.columns[1]
            print(f'{self.__class__.__name__}: second column set as item column')
        return self._item_col

    @property
    def ratings_col(self):
        assert self.columns is not None, f'{self.__class__.__name__}: columns must be assigned before calling ratings col'

        if self._ratings_col is None:
            if self.n_columns < 3:
                raise KeyError('dataset must have at least three columns for auto-setting ratings column')
            self._ratings_col = self.columns[2]
            print(f'{self.__class__.__name__}: third column set as ratings column')

    @property
    def timestamp_col(self):
        assert self.columns is not None, f'{self.__class__.__name__}: columns must be assigned before calling timestamp col'

        if self._timestamp_col is None:
            if self.n_columns < 4:
                raise KeyError('dataset must have at least four columns for auto-setting timestamp column')
            self._timestamp_col = self.columns[3]
            print(f'{self.__class__.__name__}: fourth column set as ratings column')

    def write(self):
        writer_obj = self._writer(self)
        path = writer_obj.write()
        return path

    # TODO: use writer

    def drop_column(self, column):
        if column in self.dataset.columns:
            self.dataset.drop(column, axis=1, inplace=True)
        else:
            raise KeyError('column not present in the dataset')

    # TODO: sostituire funzione info() con funzione __print__()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def users(self):
        return self._users

    @property
    def items(self):
        return self._items

    @property
    def n_users(self):
        return self._n_users

    @property
    def n_items(self):
        return self._n_items

    @property
    def transactions(self):
        return self._transactions

    @property
    def size(self):
        return self.n_items * self.n_users

    @property
    def space_size_log(self):
        return self._space_size_log

    @property
    def shape_log(self):
        return self._shape_log

    @property
    def density(self):
        return self._density

    @property
    def density_log(self):
        return self._density_log

    @property
    def gini_item(self):
        return self._gini_item

    @property
    def gini_user(self):
        return self._gini_user

    @property
    def sorted_items(self):
        return self._sorted_items

    @property
    def sorted_users(self):
        return self._sorted_users

    @property
    def user_public_to_private(self):
        if self._user_public_to_private is None:
            self._user_public_to_private = dict(zip(self.users, range(self.n_users)))
        return self._user_public_to_private

    @user_public_to_private.setter
    def user_public_to_private(self, mapping: dict):
        self._user_public_to_private = mapping

    @property
    def item_public_to_private(self):
        if self._item_public_to_private is None:
            self._item_public_to_private = dict(zip(self.items, range(self.n_items)))
        return self._item_public_to_private

    @item_public_to_private.setter
    def item_public_to_private(self, mapping: dict):
        self._item_public_to_private = mapping

    @property
    def user_private_to_public(self):
        if self._user_private_to_public is None:
            self._user_private_to_public = dict(zip(range(self.n_users), self.users))
        return self._user_private_to_public

    @user_private_to_public.setter
    def user_private_to_public(self, mapping: dict):
        self._user_private_to_public = mapping

    @property
    def item_private_to_public(self):
        if self._item_private_to_public is None:
            self._item_private_to_public = dict(zip(range(self.n_items), self.items))
        return self._item_private_to_public

    @item_private_to_public.setter
    def item_private_to_public(self, mapping: dict):
        self._item_private_to_public = mapping

    # TODO: spostare train_test_splitting in una classe splitter

    def get_metrics(self, metrics):
        result = [self.name]
        header = ['dataset'] + metrics
        for metric in metrics:
            if metric not in self._metrics:
                print(f'{self.__class__.__name__}: metric {metric} not found. Value set to None.')
            result.append(self._metrics.get(metric, None))
        return header, result

    def reset_mappings(self):
        self._user_public_to_private = None
        self._item_public_to_private = None
        self._user_private_to_public = None
        self._item_private_to_public = None

    def apply_mapping(self, column, mapping):
        return None

    def info(self):
        pass


class DPDataFrame(Dataset):

    def __init__(self, data: pd.DataFrame, path=None, data_name=None, writer=None, columns=None, **kwargs):

        assert isinstance(data, pd.DataFrame), f'{self.__class__.__name__}: data must be a pandas DataFrame'
        super().__init__(path=path, data_name=data_name, writer=writer, columns=columns)
        self.dataset = data

        self.columns = list(self.dataset.columns)

    @property
    def values(self):
        return self.dataset.values

    @property
    def users(self):
        return self.dataset[self.user_col].unique()

    @property
    def n_users(self):
        return self.dataset[self.user_col].nunique()

    @property
    def items(self):
        return self.dataset[self.item_col].unique()

    @property
    def n_items(self):
        return self.dataset[self.item_col].nunique()

    @property
    def shape(self):
        return self.n_users, self.n_items

    @property
    def transactions(self):
        if self._transactions is None:
            self._transactions = len(self.dataset)
        return self._transactions

    # METRICS
    @property
    def space_size_log(self):
        if self._space_size_log is None:
            scale_factor = 1000
            self._space_size_log = math.log10(self._n_users * self._n_items / scale_factor)
        return self._space_size_log

    @property
    def shape_log(self):
        if self._shape_log is None:
            self._shape_log = math.log10(self._n_users / self._n_items)
        return self._shape_log

    @property
    def density(self):
        if self._density is None:
            self._density = self.transactions / (self._n_users * self._n_items)
        return self._density

    @property
    def density_log(self):
        if self._density_log is None:
            self._density_log = math.log10(self.density)
        return self._density_log

    @property
    def gini_item(self):

        def gini_item_term():
            return (self._n_items + 1 - idx) / (self._n_items + 1) * self.sorted_items[item] / self._transactions

        gini_terms = 0
        for idx, (item, ratings) in enumerate(self.sorted_items.items()):
            gini_terms += gini_item_term()

        self._gini_item = 1 - 2 * gini_terms
        return self._gini_item

    @property
    def gini_user(self):

        def gini_user_term():
            return (self._n_users + 1 - idx) / (self._n_users + 1) * self.sorted_users[user] / self._transactions

        gini_terms = 0
        for idx, (user, ratings) in enumerate(self.sorted_users.items()):
            gini_terms += gini_user_term()

        self._gini_user = 1 - 2 * gini_terms
        return self._gini_user

    @property
    def sorted_items(self):
        if self._sorted_items is None:
            self._sorted_items = self.dataset.groupby(self._item_col).count() \
                .sort_values(by=[self._user_col]).to_dict()[self._user_col]
        return self._sorted_items

    @property
    def sorted_users(self):
        if self._sorted_users is None:
            self._sorted_users = self.dataset.groupby(self._user_col).count() \
                .sort_values(by=[self._item_col]).to_dict()[self._item_col]
        return self._sorted_users

    @property
    def sp_ratings(self):
        assert self.dataset is not None
        row = self.dataset[self._user_col].map(lambda x: self.user_public_to_private.get(x)).to_list()
        col = self.dataset[self._item_col].map(lambda x: self.item_public_to_private.get(x)).to_list()
        ratings = self.dataset[self._ratings_col]
        d_type = bool if self._binary else np.int8
        sp_ratings = csr_matrix((ratings, (row, col)),
                                shape=(self._n_users, self._n_items),
                                dtype=d_type)
        return sp_ratings

    @property
    def user_public_to_private(self):
        if self._user_public_to_private is None:
            self._user_public_to_private = dict(zip(self.users, range(self.n_users)))
        return self._user_public_to_private

    @user_public_to_private.setter
    def user_public_to_private(self, mapping: dict):
        self._user_public_to_private = mapping

    @property
    def item_public_to_private(self):
        if self._item_public_to_private is None:
            self._item_public_to_private = dict(zip(self.items, range(self.n_items)))
        return self._item_public_to_private

    @item_public_to_private.setter
    def item_public_to_private(self, mapping: dict):
        self._item_public_to_private = mapping

    @property
    def user_private_to_public(self):
        if self._user_private_to_public is None:
            self._user_private_to_public = dict(zip(range(self.n_users), self.users))
        return self._user_private_to_public

    @user_private_to_public.setter
    def user_private_to_public(self, mapping: dict):
        self._user_private_to_public = mapping

    @property
    def item_private_to_public(self):
        if self._item_private_to_public is None:
            self._item_private_to_public = dict(zip(range(self.n_items), self.items))
        return self._item_private_to_public

    @item_private_to_public.setter
    def item_private_to_public(self, mapping: dict):
        self._item_private_to_public = mapping

    def apply_mapping(self, column, mapping):
        assert column in self.dataset.columns, f'{self.__class__.name}: column not present in the dataset.'
        self.dataset[column] = list(map(mapping.get, self.dataset[column]))
        self.reset_mappings()

    def to_numpy(self):
        return self.dataset.values

    def to_crs(self):
        return csr_matrix(self.dataset.pivot(index=0, columns=1, values=2).fillna(0))


class DPCrsMatrix(Dataset):
    def __init__(self, data: csr_matrix, path=None, data_name=None, writer=None, columns=None, **kwargs):
        assert isinstance(data, csr_matrix), f'{self.__class__.__name__}: data must be a pandas scipy.sparse.csr_matrix'
        super().__init__(path=path, data_name=data_name, writer=writer, columns=columns)
        self.dataset = data

    @property
    def values(self):
        return self.dataset

    def __getitem__(self, key):
        return self.dataset.__getitem__(key)

    def __setitem__(self, key, value):
        self.dataset.__setitem__(key, value)

    @property
    def users(self):
        self._users = list(range(self.dataset.shape[0]))
        return self._users

    @property
    def items(self):
        self._items = list(range(self.dataset.shape[1]))
        return self._items

    @property
    def n_users(self):
        return self.dataset.shape[0]

    @property
    def n_items(self):
        return self.dataset.shape[1]

    @property
    def transactions(self):
        return len(self.dataset.data)

    @property
    def shape(self):
        return self.dataset.shape

    def user_items(self, user):
        return self.dataset[user].nonzero()[1]

    def info(self):
        print(f'data ratings: {self.transactions}')
        print(f'data users: {self.n_users}')
        print(f'data items: {self.n_items}')

    def copy_values(self):
        return DPCrsMatrix(self.dataset.copy())

    def to_numpy(self):
        return None

    def to_crs(self):
        return self.dataset


class DPDataset:

    def __init__(self, data, **kwargs):

        self.accepted_data_types = {
            pd.DataFrame: DPDataFrame,
            csr_matrix: DPCrsMatrix
        }

        data_type = type(data)
        assert data_type in self.accepted_data_types, \
            'Dataset data type not managed by this class.\n' \
            f'Found: {data_type}'\
            'Expected: \n' + '\n'.join(['\t -' + str(t) for t in self.accepted_data_types.values()])

        self._constructors = {
            pd.DataFrame: self.instantiate_dataframe,
            csr_matrix: self.instantiate_csrmatrix
        }

        self._data = self._constructors[data_type](data=data, **kwargs)

    @staticmethod
    def instantiate_dataframe(**kwargs):
        return DPDataFrame(**kwargs)

    @staticmethod
    def instantiate_csrmatrix(**kwargs):
        return DPCrsMatrix(**kwargs)

    def to_numpy(self):
        return self._data.to_numpy()

    def to_crs(self):
        return self._data.to_crs()
