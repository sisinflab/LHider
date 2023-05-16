from src.loader.paths import *
import os
import pickle
import glob

GLOBAL_SEED = 42


def experiment_info(arguments: dict):
    """
    Print information about the parameters of the experiments
    @param arguments: dictionary containing the parameters
    @return: None
    """
    for arg, value in arguments.items():
        print(f'{arg}: {value}')


def run(args: dict):
    experiment_info(args)

    # check the fundamental directories
    check_main_directories()

    # dataset directory
    dataset_name = args['dataset']

    for eps in args['eps']:
        print(f'Aggregating for eps: {eps}')

        # loading files
        eps = float(eps)

        # scores directory for the relative eps
        score_dir = scores_directory(dataset_dir=dataset_directory(dataset_name=dataset_name),
                                     eps=eps)

        # aggregate results from results directory
        aggregate_scores(score_paths=scores_files_in_directory(score_dir),
                         output_folder=score_dir,
                         delete=True)


def scores_files_in_directory(directory: str):
    """
    Returns the path of the files containing the scores in the selected directory
    @param directory: path of the directory containing scores
    @return: list of paths containing scores
    """
    # files in directory
    return glob.glob(os.path.join(directory, 'seed_*.pk'))


def aggregate_scores(score_paths: list, output_folder: str, delete: bool = False):
    """
    Given a list of paths of scores files, aggregates them and stores the aggregate result
    @param score_paths: list of score paths
    @param output_folder: path of the folder where the results will be stored
    @param delete: if true deletes aggregate files
    @return: path of the stored aggregate result
    """
    # check that score files exist:
    for score_path in score_paths:
        if not os.path.exists(score_path):
            raise FileNotFoundError(f'Score file not found at \'{score_path}\'. ')

    if len(score_paths) <= 1:
        print(f'Only one score file found: \'{score_paths[0]}\'')
        return score_paths[0]

    # check that output folder exists, otherwise create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f'Folder \'{output_folder}\' has been created')

    print('Reading score paths')
    aggregate_scores = {}
    for score_path in score_paths:
        with open(score_path, 'rb') as file:
            scores = pickle.load(file)
            aggregate_scores.update(scores)
            print(f'Scores loaded from \'{score_path}\'')

    print('Storing aggregation')
    max_seed = max(aggregate_scores.keys())
    min_seed = min(aggregate_scores.keys())
    n = len(aggregate_scores)
    output_path = os.path.join(output_folder, f'seed_{min_seed}_{max_seed}_n{n}.pk')

    with open(output_path, 'wb') as file:
        pickle.dump(aggregate_scores, file)
    print(f'Aggregated scores stored at \'{output_path}\'')

    # remove old score files
    if delete:
        for score_path in score_paths:
            os.remove(score_path)

    return output_path

