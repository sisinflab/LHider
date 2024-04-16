from data_preprocessing.filters.filter import store_dataset
from src.loader.paths import *
from sklearn.model_selection import train_test_split as split


import pandas as pd


def run(dataset_name, numeric_to_categorical_columns=None, as_category_columns=None, drop_columns=None, categorical_to_one_hot=True, drop_na_value=None, sep=',', remove_last_column=False):
    print(f'\n***** {dataset_name} data preprocessing *****\n'.upper())

    dataset_path = dataset_filepath(dataset_name, type='raw')
    dataset = pd.read_csv(dataset_path, header=None, sep=sep)
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

    if remove_last_column:
        dataset = dataset.iloc[:,:-1]

    # SPLITTING
    print(f'\n***** {dataset_name} train-test splitting *****\n'.upper())

    print(f'\nTransactions: {len(dataset)}')
    print('\nThere will be the splitting...')

    train_set, test_set = split(dataset, test_size=0.2, random_state=42)

    splitting_results = {}
    splitting_results['train'] = train_set
    splitting_results['test'] = test_set

    print(f'Final training set transactions: {len(splitting_results["train"])}')
    print(f'Final test set transactions: {len(splitting_results["test"])}')

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
