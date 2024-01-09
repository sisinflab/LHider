import argparse
from utils.recs_best_model import run
import os
import pandas as pd
from src.loader.paths import *
from data_preprocessing.filters.dataset import Splitter
from data_preprocessing.filters.filter import load_dataset, store_dataset
from src.loader.loaders import TsvLoader
from src.dataset.dataset import *

DEFAULT_METRICS = ["nDCGRendle2020", "Recall", "HR", "nDCG", "Precision", "F1", "MAP", "MAR", "ItemCoverage", "Gini",
                   "SEntropy", "EFD", "EPC", "PopREO", "PopRSP", "ACLT", "APLT", "ARP"]

# dataset
dataset_name = 'facebook_books'
dataset_type = 'train'
result_dir = os.path.join(RESULT_DIR, 'perturbed_datasets', dataset_name + '_' + dataset_type)

files = os.listdir(result_dir)



# loading test file
test_path = dataset_filepath(dataset_name, 'test')
loader = TsvLoader(path=test_path, return_type="csr")
test = DPCrsMatrix(loader.load(), path=test_path, data_name=dataset_name)

from src.jobs.sigir import from_csr_to_pandas

test = from_csr_to_pandas(csr_matrix(test.dataset))
test['r'] = 1

generated_folder = os.path.join(dataset_directory(dataset_name),
                                'generated_' + dataset_type)
if os.path.exists(generated_folder) is False:
    os.makedirs(generated_folder)
    print(f'created directory \'{generated_folder}\'')

store_dataset(data=test,
              folder=generated_folder,
              name='test',
              message='test set')

for file in files:
    if '.tsv' in file:
        data_name = file.replace('.tsv', '')
        dataset_path = os.path.join(result_dir, file)

        dataset = pd.read_csv(dataset_path, sep='\t', header=None, names=['u', 'i'])

        splitter = Splitter(data=dataset,
                            test_ratio=0.2)
        splitting_results = splitter.filter()

        data_folder = os.path.join(generated_folder,
                                   data_name)
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
