import pandas as pd
from src.loader.paths import *
from data_preprocessing.filters.dataset import Splitter
from data_preprocessing.filters.filter import load_dataset, store_dataset
from elliot.run import run_experiment
from config_templates.short_training import TEMPLATE_PATH

DEFAULT_METRICS = ["nDCGRendle2020", "Recall", "HR", "nDCG", "Precision", "F1", "MAP", "MAR", "ItemCoverage", "Gini",
                   "SEntropy", "EFD", "EPC", "PopREO", "PopRSP", "ACLT", "APLT", "ARP"]

# dataset
dataset_name = 'yahoo_movies'
dataset_type = 'train'

dataset_path = dataset_filepath(dataset_name, dataset_type)

dataset = pd.read_csv(dataset_path, sep='\t', header=None, names=['u', 'i', 'r'])

splitter = Splitter(data=dataset,
                    test_ratio=0.2)
splitting_results = splitter.filter()

data_folder = os.path.join(dataset_directory(dataset_name),
                           'generated_' + dataset_type, 'original')
if os.path.exists(data_folder) is False:
    os.makedirs(data_folder)
    print(f'created directory \'{data_folder}\'')

train = splitting_results["train"]
val = splitting_results["test"]

from src.dataset.dataset import *
from src.loader.loaders import *
from src.jobs.sigir import from_csr_to_pandas

train['r'] = 1
val['r'] = 1

train_path = store_dataset(data=splitting_results["train"],
              folder=data_folder,
              name='train',
              message='training set')['train']

val_path = store_dataset(data=splitting_results["test"],
              folder=data_folder,
              name='validation',
              message='val set')['validation']

loader = TsvLoader(path=train_path, return_type="csr")
train = DPCrsMatrix(loader.load(), path=train_path, data_name=dataset_name)
train = from_csr_to_pandas(csr_matrix(train.dataset))

loader = TsvLoader(path=val_path, return_type="csr")
val = DPCrsMatrix(loader.load(), path=val_path, data_name=dataset_name)
val = from_csr_to_pandas(csr_matrix(val.dataset))

train['r'] = 1
val['r'] = 1

train_path = store_dataset(data=train,
              folder=data_folder,
              name='train',
              message='training set')['train']

val_path = store_dataset(data=val,
              folder=data_folder,
              name='validation',
              message='val set')['validation']

train_path = os.path.join(data_folder, 'train.tsv')
val_path = os.path.join(data_folder, 'validation.tsv')
test_path = os.path.join(dataset_directory(dataset_name),
                         'generated_' + dataset_type,
                         'test.tsv')

output_folder = os.path.join(PROJECT_PATH, 'results_collection')
if os.path.exists(output_folder) is False:
    os.makedirs(output_folder)
    print(f'Created folder at \'{output_folder}\'')

data_name = 'original'

assert os.path.exists(train_path)
assert os.path.exists(val_path)
assert os.path.exists(test_path)

result_folder = os.path.join(output_folder, data_name)
if os.path.exists(result_folder) is False:
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