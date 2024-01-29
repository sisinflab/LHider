import pandas as pd
from src.loader.paths import *


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
    seed = int(params[4])
    total_eps = float(params[5])
    return randomizer, eps_phi, reps, eps_exp, seed, total_eps


MODELS = {
    'EASER': 'EASEr',
    'MostPop': 'MP',
    'ItemKNN': 'ItemKNN'
}


def run(args):

    dataset_name = args['dataset_name']
    dataset_type = args['dataset_type']
    base_seed = args['base_seed']

    result_dir = os.path.join(PROJECT_PATH, 'results_collection', dataset_name + '_' + dataset_type, str(base_seed))
    # result_dir = os.path.join(PROJECT_PATH, 'generated_datasets/facebook_books_train/3.0_1.0_42/recs')

    result = None
    for dataset in os.listdir(result_dir):
        if dataset == '.DS_Store':
            continue
        dataset_dir = os.path.join(result_dir, dataset)
        perf_file = [x for x in os.listdir(dataset_dir) if 'rec_cutoff' in x]
        if len(perf_file) == 0:
            print(f'OCCHIO A {dataset}')
        perf_file = perf_file[0]
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
        #params = [os.path.basename(dataset_dir).split('_')[-1], 3.0, 1, 1.0]
        row_res = [params + list(vals.values())]

        columns = ['method', 'eps_phi', 'n', 'eps_exp', 'seed', 'total_eps'] + list(vals.keys())
        row_vals = pd.DataFrame(row_res, columns=columns)

        if result is None:
            result = row_vals
        else:
            result = pd.concat([result, row_vals])

    # save
    output_dir = os.path.join(PROJECT_PATH, 'results_data', dataset_name + '_' + dataset_type, str(base_seed))
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
        print(f'Created directory {output_dir}')

    output_path = os.path.join(output_dir, 'aggregated_results.tsv')
    result.to_csv(output_path, sep='\t', header=True, index=False, decimal=',')
    print(f'Results stored at \'{output_path}\'')
