import os
from src.loader.paths import *
import pandas as pd

dataset_name = 'yahoo_movies'
dataset_type = 'train'

results = None

for base_seed in range(100, 1001, 100):

    results_dir = os.path.join(PROJECT_PATH, 'results_data', dataset_name + '_' + dataset_type, str(base_seed))
    results_path = os.path.join(results_dir, 'aggregated_results.tsv')

    data = pd.read_csv(results_path, sep='\t')
    if results is None:
        results = data
    else:
        results = pd.concat([results, data])

output_path = os.path.join(PROJECT_PATH, 'results_data', dataset_name + '_' + dataset_type, 'final_results.tsv')
results.to_csv(output_path, sep='\t', index=False, decimal=',')
print(f'Results stored at \'{output_path}\'')



