import os.path
# from data_preprocessing.filters.dataset import Binarize, Splitter
from data_preprocessing.filters.kcore import IterativeKCore
from data_preprocessing.filters.dataset import load_dataset, store_dataset

dataset_relative_path = 'dataset.tsv'
dataset_dir = os.path.join(os.path.dirname(os.path.abspath(os.curdir)), "data", "yahoo_movies")
dataset_path = os.path.join(dataset_dir, dataset_relative_path)

dataset = load_dataset(dataset_path)

kcore = IterativeKCore(dataset, ['u', 'i', 'r'], 5)

filtered_dataset = kcore.filter()

print()

# def run(data_folder):
#     data_name = os.path.split(data_folder)
#
#     print(f'\n***** {data_name} data preparation *****\n'.upper())
#     dataset_path = os.path.join(data_folder, dataset_relative_path)
#     dataset = load_dataset(dataset_path)
#
#     print(f'\nTransactions: {len(dataset)}')
#     print('\nThere will be the splitting...')
#
#     splitter = Splitter(data=dataset,
#                         test_ratio=0.2,
#                         val_ratio=0.1)
#
#     splitting_results = splitter.filter()
#     print(f'Final training set transactions: {len(splitting_results["train"])}')
#     print(f'Final test set transactions: {len(splitting_results["test"])}')
#     print(f'Final validation set transactions: {len(splitting_results["val"])}')
#
#     store_dataset(data=splitting_results["train"],
#                   folder=data_folder,
#                   name='train',
#                   message='training set')
#
#     store_dataset(data=splitting_results["test"],
#                   folder=data_folder,
#                   name='test',
#                   message='test set')
#
#     store_dataset(data=splitting_results["val"],
#                   folder=data_folder,
#                   name='val',
#                   message='validation set')


