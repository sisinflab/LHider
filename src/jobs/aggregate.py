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


def run(args):
    # print information about the experiment
    experiment_info(args)

    # check the fundamental directories
    check_main_directories()

    # loading files
    dataset_name = args['dataset']

    # dataset result directory
    dataset_result_dir = os.path.join(RESULT_DIR, dataset_name)

    # aggregate results from results directory
    aggregate_results(dataset_result_dir)


def aggregate_results(folder):

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
    result_path = os.path.join(aggregate_dir, f'seed_score_{min_seed}_{max_seed}.pk')
    with open(result_path, 'wb') as result_file:
        pickle.dump(results, result_file)
    print(f'aggregation written at \'{result_path}\'')
