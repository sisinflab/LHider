from data_preprocessing.filters.dataset import Splitter
from data_preprocessing.filters.filter import store_dataset
from src.loader.paths import *

import pandas as pd


def run(dataset_name, numeric_to_categorical_columns=None, as_category_columns=None, drop_columns=None, categorical_to_one_hot=True, drop_na_value=None):
    print(f'\n***** {dataset_name} data preprocessing *****\n'.upper())

    dataset_path = dataset_filepath(dataset_name, type='raw')
    dataset = pd.read_csv(dataset_path, header=None, sep=',')
    print(f'Dataset loaded from {dataset_path}')

    if drop_na_value:
        dataset = dataset.replace(' ?', pd.NA).dropna()

    if numeric_to_categorical_columns:
        for i in numeric_to_categorical_columns:
            dataset[i] = pd.cut(dataset[i], 4, labels=[0, 1, 2, 3])

    if drop_columns:
        dataset = dataset.drop(columns=drop_columns)

    if as_category_columns:
        for column in as_category_columns:
            dataset[column] = dataset[column].astype('category')

    if categorical_to_one_hot:
        dataset = pd.get_dummies(dataset)

    # SPLITTING
    print(f'\n***** {dataset_name} train-test splitting *****\n'.upper())

    print(f'\nTransactions: {len(dataset)}')
    print('\nThere will be the splitting...')

    splitter = Splitter(data=dataset,
                        test_ratio=0.2)

    splitting_results = splitter.filter()
    print(f'Final training set transactions: {len(splitting_results["train"])}')
    print(f'Final test set transactions: {len(splitting_results["test"])}')
    print(f'Final validation set transactions: {len(splitting_results["val"])}')

    # STORE ON FILE
    data_folder = dataset_directory(dataset_name=dataset_name)

    store_dataset(data=dataset,
                  folder=data_folder,
                  name='dataset',
                  message='filtered dataset')

    store_dataset(data=splitting_results["train"],
                  folder=data_folder,
                  name='train',
                  message='training set')

    store_dataset(data=splitting_results["test"],
                  folder=data_folder,
                  name='test',
                  message='test set')
