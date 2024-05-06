import argparse
import os.path

from config_templates.training import TEMPLATE_PATH
from elliot.run import run_experiment
from data_preprocessing.filters.dataset import Splitter
import pandas as pd
from src.loader.paths import *
from data_preprocessing.filters.filter import store_dataset


def run(dataset_name):
    dataset_folder = os.path.join(PROJECT_PATH, 'data', dataset_name)
    dataset = pd.read_csv(os.path.join(dataset_folder, 'train.tsv'), sep='\t', header=None, names=['u', 'i', 'r'])

    splitter = Splitter(data=dataset, test_ratio=0.2)

    splitting_results = splitter.filter()

    result_folder = os.path.join(PROJECT_PATH, 'data', dataset_name, 'baseline')
    os.makedirs(result_folder, exist_ok=True)
    train_path = os.path.join(result_folder, 'train.tsv')
    val_path = os.path.join(result_folder, 'validation.tsv')
    test_path = os.path.join(dataset_folder, 'test.tsv')

    train = splitting_results["train"]
    val = splitting_results["test"]

    train['r'] = 1
    val['r'] = 1

    store_dataset(data=splitting_results["train"],
                  folder=result_folder,
                  name='train',
                  message='training set')

    store_dataset(data=splitting_results["test"],
                  folder=result_folder,
                  name='validation',
                  message='val set')

    config = TEMPLATE_PATH.format(dataset=dataset_name,
                                  output_path=result_folder,
                                  train_path=train_path,
                                  val_path=val_path,
                                  test_path=test_path)

    config_path = os.path.join(CONFIG_DIR, 'runtime.yml')
    with open(config_path, 'w') as conf_file:
        conf_file.write(config)

    run_experiment(config_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = vars(parser.parse_args())
    run(args['dataset'])

