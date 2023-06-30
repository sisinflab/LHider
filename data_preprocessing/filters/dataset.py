import pandas as pd
import os


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
