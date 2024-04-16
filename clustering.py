import argparse
import os.path

import numpy as np
import pandas as pd
import tqdm
import sklearn.metrics
from sklearn.cluster import KMeans

from data_preprocessing.filters.dataset import Splitter
from src.loader.paths import *
from data_preprocessing.filters.filter import store_dataset
from utils.collect_results import run as run_collect
from src.jobs.generate import run_generation


n = 100
folder = 0
seed = 0


class ClusteringExperiment():
    def __init__(self):
        self.clf = KMeans(random_state=0, cv=2, class_weight='balanced')

    def train(self, features):
        self.clf.fit(features)

    def score(self, features, labels, train_data):
        #TODO: Vedere un po' qua
        predicted_values = self.clf.predict(features)
        values = {'model': 'LogReg', 'F1': sklearn.metrics.f1_score(predicted_values, labels),
                          'Accuracy': sklearn.metrics.accuracy_score(predicted_values, labels),
                           'Recall': sklearn.metrics.recall_score(predicted_values, labels),
                  'ZeroOneUtility': - sklearn.metrics.zero_one_loss(self.clf.predict(train_data[0]), train_data[1])}
        return pd.DataFrame(values, index=[0])



def run_baseline(dataset_name):
    dataset_folder = os.path.join(PROJECT_PATH, 'data', dataset_name)
    dataset = pd.read_csv(os.path.join(dataset_folder, 'train.tsv'), sep='\t', header=None)

    splitter = Splitter(data=dataset, test_ratio=0.2)

    splitting_results = splitter.filter()

    result_folder = os.path.join(PROJECT_PATH, 'data', dataset_name, 'baseline')
    os.makedirs(result_folder, exist_ok=True)
    train_path = os.path.join(result_folder, 'train.tsv')
    val_path = os.path.join(result_folder, 'validation.tsv')
    test_path = os.path.join(dataset_folder, 'test.tsv')

    train = splitting_results["train"]
    val = splitting_results["test"]

    store_dataset(data=splitting_results["train"],
                  folder=result_folder,
                  name='train',
                  message='training set')

    store_dataset(data=splitting_results["test"],
                  folder=result_folder,
                  name='validation',
                  message='val set')

    x_train = train.drop(columns=[len(train.columns) - 1])
    y_train = train[len(train.columns) - 1]

    x_test = val.drop(columns=[len(val.columns) - 1])
    y_test = val[len(val.columns) - 1]

    clustering = ClusteringExperiment()
    clustering.train(train)
    clustering.score(val, train)


def run_clustering(args):
    dataset_name = args['dataset_name']
    dataset_type = args['dataset_type']
    base_seed = args['base_seed']
    from src.jobs.sigir import noisy_dataset_folder

    result_dir = noisy_dataset_folder(dataset_name=dataset_name,
                                      dataset_type=dataset_type,
                                      base_seed=base_seed)

    files = os.listdir(result_dir)

    test_path = dataset_filepath(dataset_name, 'test')
    test = pd.read_csv(test_path, sep='\t', header=None)

    output_folder = os.path.join(PROJECT_PATH, 'results_collection', dataset_name + '_' + dataset_type, str(base_seed))
    if os.path.exists(output_folder) is False:
        os.makedirs(output_folder)
        print(f'Created folder at \'{output_folder}\'')
    print(f'Results will be stored at \'{output_folder}\'')

    for file in tqdm.tqdm(files):
        if '.tsv' in file:
            data_name = file.replace('.tsv', '')

            try:
                result_folder = os.path.join(output_folder, data_name)
                os.makedirs(result_folder, exist_ok=True)

                dataset = pd.read_csv(os.path.join(result_dir, file), sep='\t', header=None)

                x_train = dataset.drop(columns=[len(dataset.columns) - 1])
                y_train = dataset[len(dataset.columns) - 1]

                x_test = test.drop(columns=[len(test.columns) - 1])
                y_test = test[len(test.columns) - 1]

                logistic_regression = ClusteringExperiment()
                logistic_regression.train(x_train, y_train)

                results = logistic_regression.score(x_test, y_test, (x_train, y_train))
                results.to_csv(os.path.join(result_folder, 'rec_cutoff_model.tsv'), index=False, sep='\t')


            except Exception as e:
                print(e)
                print(f'Keep attention: recommendation has not been computer for {dataset_name}')





def run(args):
    for eph_phi in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
        args = {
            'dataset': args['dataset'],
            'dataset_name': args['dataset'],
            'dataset_type': 'train',
            'type': 'train',
            'eps_phi': eph_phi,
            'randomizer': args['randomizer'],
            'base_seed': folder,
            'score_type': args['score_type'],
            'generations': n,
            'seed': seed,
            'return_type': 'sparse_ml',
            'ml_task': True
        }
        run_generation(args)
    # run_split(args)
    run_clustering(args)
    run_collect(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--score_type', default='euclidean')
    parser.add_argument('--randomizer', default='randomized')
    arguments = parser.parse_args()
    run(vars(arguments))
