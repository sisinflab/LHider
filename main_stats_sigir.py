from src.loader.paths import RESULT_DIR
import os
import pandas as pd
from src.dataset.dataset import DPDataFrame

def params_from_name(file_name: str):
    file_name = file_name.replace('.tsv', '')
    params = file_name.split('_')
    randomizer = params[0]
    eps_phi = float(params[1])
    reps = int(params[2])
    eps_exp = float(params[3])
    return randomizer, eps_phi, reps, eps_exp

def load_data():
    pass

# dataset
dataset_name = 'facebook_books'
dataset_type = 'raw'
result_dir = os.path.join(RESULT_DIR, 'perturbed_datasets', dataset_name + '_' + dataset_type)

files = os.listdir(result_dir)

result = []
characteristics = ['size', 'transactions', 'density', 'gini_item', 'gini_user']

for file in files:
    if '.tsv' in file:
        file_path = os.path.join(result_dir, file)
        data = pd.read_csv(file_path, sep='\t', header=None)
        print(f'reading file: \'{file_path}\'')

        randomizer, eps_phi, reps, eps_exp = params_from_name(file)

        dataa = DPDataFrame(data, path=file_path, data_name=file)

        result.append([dataset_name, dataset_type, randomizer, eps_phi, reps, eps_exp] +
                      [dataa.size, dataa.transactions, dataa.density, dataa.gini_item, dataa.gini_user])

header = ['dataset', 'type', 'randomizer', 'eps_phi', 'n', 'eps_exp'] + characteristics

characteristics_dir = os.path.join(result_dir, 'characteristics')
if not(os.path.exists(characteristics_dir)):
    os.makedirs(characteristics_dir)
    print(f'Directory created at: \'{characteristics_dir}\'')
dataframe = pd.DataFrame(result, columns=header)
result_path = os.path.join(characteristics_dir, 'characteristics.tsv')
dataframe.to_csv(result_path, sep='\t', index=False)
print(f'Characteristics stored at \'{result_path}\'')
print()
