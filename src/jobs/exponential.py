import os.path

from src.loader.paths import *
from src.loader import *
from src.exponential_mechanism import *
from src.laplace_mechanism.mechanism import *


METRICS = ['method', 'eps_phi', 'n', 'eps_exp', 'seed', 'total_eps',
       'MP_nDCGRendle2020', 'MP_Recall', 'MP_HR', 'MP_nDCG', 'MP_Precision',
       'MP_F1', 'MP_MAP', 'MP_MAR', 'MP_LAUC', 'MP_ItemCoverage', 'MP_Gini',
       'MP_SEntropy', 'MP_EFD', 'MP_EPC', 'MP_PopREO', 'MP_PopRSP', 'MP_ACLT',
       'MP_APLT', 'MP_ARP', 'EASEr_nDCGRendle2020', 'EASEr_Recall', 'EASEr_HR',
       'EASEr_nDCG', 'EASEr_Precision', 'EASEr_F1', 'EASEr_MAP', 'EASEr_MAR',
       'EASEr_LAUC', 'EASEr_ItemCoverage', 'EASEr_Gini', 'EASEr_SEntropy',
       'EASEr_EFD', 'EASEr_EPC', 'EASEr_PopREO', 'EASEr_PopRSP', 'EASEr_ACLT',
       'EASEr_APLT', 'EASEr_ARP', 'ItemKNN_nDCGRendle2020', 'ItemKNN_Recall',
       'ItemKNN_HR', 'ItemKNN_nDCG', 'ItemKNN_Precision', 'ItemKNN_F1',
       'ItemKNN_MAP', 'ItemKNN_MAR', 'ItemKNN_LAUC', 'ItemKNN_ItemCoverage',
       'ItemKNN_Gini', 'ItemKNN_SEntropy', 'ItemKNN_EFD', 'ItemKNN_EPC',
       'ItemKNN_PopREO', 'ItemKNN_PopRSP', 'ItemKNN_ACLT', 'ItemKNN_APLT',
       'ItemKNN_ARP']

metric = 'MP_nDCGRendle2020'


def run(args):
    performance_file = os.path.join(PROJECT_PATH, 'results_data',
                                    args['dataset_name'] + '_' + args['dataset_type'], str(args['base_seed']),
                                    'aggregated_results.tsv')

    performance = pd.read_csv(performance_file, sep='\t', decimal=',', header=0)

    directory = os.path.join(PROJECT_PATH, 'results_collection',
                             args['dataset_name'] + '_' + args['dataset_type'], str(args['base_seed']))

    seeds = []
    scores = []
    total_scores = dict()

    # loading files
    dataset_path = dataset_filepath(args['dataset_name'], args['dataset_type'])
    loader = TsvLoader(path=dataset_path, return_type="sparse")
    data = loader.load().A

    for file_name in os.listdir(directory):
        params = dict(zip(['method', 'eps_z', 'reps', 'score', 'seed', 'total_eps'], file_name.split('_')))

        eps_z = float(params['eps_z'])

        file_seed = int(params['seed'])
        file_score = float(params['score'])

        if total_scores.get(eps_z) is None:
            total_scores[eps_z] = {
                'seeds': [file_seed],
                'scores': [file_score]
            }
        else:
            total_scores[eps_z]['seeds'].append(file_seed)
            total_scores[eps_z]['scores'].append(file_score)

    eps_exponentials = [0.001, 0.01, 0.1, 1, 2, 5, 10, 100]
    exponential_random_seed = 0

    global_results = []

    for esp_z in total_scores:
        cols = ['eps_z']
        row = [esp_z]

        performance_eps_z = performance[performance.eps_phi == esp_z]

        z_seeds = total_scores[esp_z]['seeds']
        z_scores = total_scores[esp_z]['scores']
        values = list(zip(z_seeds, z_scores))

        max_scores = performance_eps_z[metric].max()
        mean_scores = performance_eps_z[metric].mean()
        min_scores = performance_eps_z[metric].min()

        row.append(max_scores)
        row.append(mean_scores)
        row.append(min_scores)

        cols.append('global_max')
        cols.append('global_mean')
        cols.append('global_min')

        for dim in args['dimensions']:
            if dim > len(values):
                dim = len(values)

            exp_results = []
            for eps_exp in eps_exponentials:
                for trial in range(10):
                    idx = list(range(len(values)))
                    sampled_idx = np.random.choice(idx, dim, replace=False)
                    samples = np.array(values)[sampled_idx]

                    sampled_seeds = samples[:, 0]
                    sampled_scores = samples[:, 1]

                    exponential_random_seed += 1
                    exp_mech = ExponentialMechanism(JaccardDistance(data), eps_exp, exponential_random_seed)
                    output = exp_mech.run_exponential_sensibile(sampled_seeds, np.array(sampled_scores))
                    exp_results.append(performance_eps_z[performance_eps_z.seed == int(output)][metric].values[0])

                row.append(np.max(exp_results))
                row.append(np.mean(exp_results))
                row.append(np.min(exp_results))
                cols.append(f'max_{dim}_{eps_exp}')
                cols.append(f'mean_{dim}_{eps_exp}')
                cols.append(f'min_{dim}_{eps_exp}')
        global_results.append(row)

    dataframe = pd.DataFrame(global_results, columns=cols).sort_values(by=['eps_z'])
    path = os.path.join(PROJECT_PATH, 'results_data',
                        args['dataset_name'] + '_' + args['dataset_type'], str(args['base_seed']),
                        f'{args["dataset_name"]}_{args["dataset_type"]}_{metric}.tsv')
    dataframe.to_csv(path, sep='\t', index=False)
    print(f'results stored at \'{path}\'')
