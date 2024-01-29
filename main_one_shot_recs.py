from elliot.run import run_experiment
from data_preprocessing.filters.dataset import Splitter
import pandas as pd
import os
from split_generated import run as run_split
from src.loader.paths import *
from data_preprocessing.filters.filter import store_dataset


dataset_path = '/home/alberto/PycharmProjects/LHider/data/facebook_books/train.tsv'
dataset = pd.read_csv(dataset_path, sep='\t', header=None, names=['u', 'i', 'r'])

splitter = Splitter(data=dataset, test_ratio=0.2)

splitting_results = splitter.filter()

data_folder = '/home/alberto/PycharmProjects/LHider/data/facebook_books/baseline'
if os.path.exists(data_folder) is False:
    os.makedirs(data_folder)
    print(f'created directory \'{data_folder}\'')

train = splitting_results["train"]
val = splitting_results["test"]

train['r'] = 1
val['r'] = 1

store_dataset(data=splitting_results["train"],
              folder=data_folder,
              name='train',
              message='training set')

store_dataset(data=splitting_results["test"],
              folder=data_folder,
              name='validation',
              message='val set')


run_experiment('/home/alberto/PycharmProjects/LHider/config_files/yahoo_baseline.yml')
