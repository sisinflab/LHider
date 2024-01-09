import os.path
import pandas as pd
from src.loader.paths import *

MODELS = {
    'EASER': 'EASEr',
    'MostPop': 'MP'
}

result_dir = os.path.join(PROJECT_PATH, 'results_collection')

def change_model(name):
    for m in MODELS:
        if m in name:
            return MODELS[m]

def params_from_name(file_name: str):
    file_name = file_name.replace('.tsv', '')
    params = file_name.split('_')
    randomizer = params[0]
    eps_phi = float(params[1])
    reps = int(params[2])
    eps_exp = float(params[3])
    return randomizer, eps_phi, reps, eps_exp


result = None
for dataset in os.listdir(result_dir):
    dataset_dir = os.path.join(result_dir, dataset)
    perf_file = [x for x in os.listdir(dataset_dir) if 'rec_cutoff' in x][0]
    perf_path = os.path.join(dataset_dir, perf_file)

    perf = pd.read_csv(perf_path, sep='\t', header=0)
    perf.model = perf.model.apply(lambda x: change_model(x))
    cols = list(perf.columns)

    vals = {}
    for row in perf.values:

        model = row[0]
        for metric_value, metric_name in zip(row[1:], cols[1:]):
            vals[model + '_' + metric_name] = metric_value

    params = list(params_from_name(dataset))
    row_res = [params + list(vals.values())]

    columns = ['method', 'eps_phi', 'n', 'eps_exp'] + list(vals.keys())
    row_vals = pd.DataFrame(row_res, columns=columns)

    if result is None:
        result = row_vals
    else:
        result = pd.concat([result, row_vals])

print()
# save
output_path = os.path.join(RESULT_DIR, 'aggregated_results.tsv')
result.to_csv(output_path, sep='\t', header=True, index=False)
print(f'Results stored at \'{output_path}\'')
