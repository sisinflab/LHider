import argparse
from utils.recs_best_model import run
import os
import pandas as pd
from src.loader.paths import *
from data_preprocessing.filters.dataset import Splitter
from data_preprocessing.filters.filter import load_dataset, store_dataset
from elliot.run import run_experiment
from config_templates.short_training import TEMPLATE_PATH
import tqdm

DEFAULT_METRICS = ["nDCGRendle2020", "Recall", "HR", "nDCG", "Precision", "F1", "MAP", "MAR", "ItemCoverage", "Gini",
                   "SEntropy", "EFD", "EPC", "PopREO", "PopRSP", "ACLT", "APLT", "ARP"]

# dataset
dataset_name = 'facebook_books'
dataset_type = 'train'
result_dir = os.path.join(RESULT_DIR, 'perturbed_datasets', dataset_name + '_' + dataset_type)

files = os.listdir(result_dir)

test_path = os.path.join(dataset_directory(dataset_name),
                                   'generated_' + dataset_type,
                                   'test.tsv')

output_folder = os.path.join(PROJECT_PATH, 'results_collection')
if os.path.exists(output_folder) is False:
    os.makedirs(output_folder)
    print(f'Created folder at \'{output_folder}\'')


for file in tqdm.tqdm(files):
    if '.tsv' in file:
        data_name = file.replace('.tsv', '')

        data_folder = os.path.join(dataset_directory(dataset_name),
                                   'generated_' + dataset_type,
                                   data_name)
        train_path = os.path.join(data_folder, 'train.tsv')
        val_path = os.path.join(data_folder, 'validation.tsv')

        assert os.path.exists(train_path)
        assert os.path.exists(val_path)
        assert os.path.exists(test_path)

        result_folder = os.path.join(output_folder, data_name)
        os.makedirs(result_folder)

        config = TEMPLATE_PATH.format(dataset=data_name,
                                      output_path=result_folder,
                                      train_path=train_path,
                                      val_path=val_path,
                                      test_path=test_path)


        config_path = os.path.join(CONFIG_DIR, 'runtime.yml')
        with open(config_path, 'w') as conf_file:
            conf_file.write(config)

        run_experiment(config_path)
