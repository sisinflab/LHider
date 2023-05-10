from src.loader.paths import *
import os
import pickle

GLOBAL_SEED = 42


def experiment_info(arguments: dict):
    """
    Print information about the parameters of the experiments
    @param arguments: dictionary containing the paramenters
    @return: None
    """
    for arg, value in arguments.items():
        print(f'{arg}: {value}')


def run(args: dict):
    # print information about the experiment
    experiment_info(args)

    # check the fundamental directories
    check_main_directories()

    # loading files
    dataset_name = args['dataset']
    output_name = args['output']

    # dataset result directory
    dataset_result_dir = os.path.join(RESULT_DIR, dataset_name)

    # aggregate results from results directory
    aggregate_results(dataset_result_dir, output_name)


def aggregate_results(folder: str, output_name: str = None):
    print('reading paths')
    aggregate_dir = os.path.join(folder, 'aggregate')
    if not os.path.exists(aggregate_dir):
        os.makedirs(aggregate_dir)
    folders = [os.path.join(folder, f) for f in os.listdir(folder)]
    if aggregate_dir in folders:
        folders.remove(aggregate_dir)

    print('files loading and aggregation')
    files = [os.path.join(f, file) for f in folders for file in os.listdir(f)]
    results = dict()
    for path in files:
        print(f'Reading: \'{path}\'')
        with open(path, 'rb') as file:
            local_result = pickle.load(file)
        results.update(local_result)

    print('storing aggregation')
    max_seed = max(results.keys())
    min_seed = min(results.keys())

    if output_name:
        result_path = os.path.join(aggregate_dir, f'{output_name}.pk')
    else:
        result_path = os.path.join(aggregate_dir, f'seed_score_{min_seed}_{max_seed}.pk')

    with open(result_path, 'wb') as result_file:
        pickle.dump(results, result_file)
    print(f'aggregation written at \'{result_path}\'')


def aggregate_scores(score_paths: list, output_folder: str):
    print('reading score paths')
    aggregate_scores = {}
    for score_path in score_paths:
        with open(score_path, 'rb') as file:
            scores = pickle.load(file)
            aggregate_scores.update(scores)
            print(f'scores loaded from \'{score_path}\'')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f'Folder \'{output_folder}\' has been created')

    print('storing aggregation')
    max_seed = max(aggregate_scores.keys())
    min_seed = min(aggregate_scores.keys())
    n = len(aggregate_scores)

    output_path = os.path.join(output_folder, f'seed_{min_seed}_{max_seed}_n{n}.pk')

    with open(output_path, 'wb') as file:
        pickle.dump(aggregate_scores, file)

    print(f'Aggregated scores stored at \'{output_path}\'')
