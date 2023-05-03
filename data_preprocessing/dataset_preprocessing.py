import os.path
from data_preprocessing.filters.dataset import Binarize, Splitter
from data_preprocessing.filters import load_kg, load_dataset, load_linking, store_dataset, store_mapped_kg

dataset_relative_path = 'dataset.tsv'


def run(data_folder):
    data_name = os.path.split(data_folder)

    print(f'\n***** {data_name} data preparation *****\n'.upper())
    dataset_path = os.path.join(data_folder, dataset_relative_path)
    dataset = load_dataset(dataset_path)

    print(f'\nTransactions: {len(dataset)}')
    print('\nThere will be the splitting...')

    splitter = Splitter(data=dataset,
                        test_ratio=0.2,
                        val_ratio=0.1)

    splitting_results = splitter.filter()
    print(f'Final training set transactions: {len(splitting_results["train"])}')
    print(f'Final test set transactions: {len(splitting_results["test"])}')
    print(f'Final validation set transactions: {len(splitting_results["val"])}')

    store_dataset(data=splitting_results["train"],
                  folder=data_folder,
                  name='train',
                  message='training set')

    store_dataset(data=splitting_results["test"],
                  folder=data_folder,
                  name='test',
                  message='test set')

    store_dataset(data=splitting_results["val"],
                  folder=data_folder,
                  name='val',
                  message='validation set')
