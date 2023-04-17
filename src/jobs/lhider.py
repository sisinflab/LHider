from src.loader import *
from src.loader.paths import *
from src.dataset.dataset import DPCrsMatrix
from src.exponential_mechanism.mechanism import ExponentialMechanism
from src.exponential_mechanism.scores import MatrixCosineSimilarity, LoadScores
from src.randomize_response.mechanism import RandomizeResponse
import pandas as pd
import os

GLOBAL_SEED = 42


def experiment_info(arguments: dict):
    """
    Print information about the parameters of the experiments
    @param arguments: dictionary containing the parameters
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
    dataset_path = args['dataset']
    loader = TsvLoader(path=dataset_path, return_type="csr")
    data = DPCrsMatrix(loader.load(), path=dataset_path)

    # dataset result directory
    dataset_result_dir = os.path.join(RESULT_DIR, data.name)
    create_directory(dataset_result_dir)

    # print dataset info
    # TODO: implementare qui il metodo info della classe dataset
    print(f'data ratings: {data.transactions}')
    print(f'data users: {data.n_users}')
    print(f'data items: {data.n_items}')

    scores_name = args['scores']
    scores_path = os.path.join(dataset_result_dir, scores_name)
    eps = args['eps']
    change_prob = args['change_prob']

    for eps_ in eps:
        c_seed, c_score = exponential_mechanism(scores_path=scores_path, eps=eps_)
        print(f'Selected a dataset with score: {c_score}')
        gen_dataset = generate(data=data, change_probability=change_prob, seed=c_seed-GLOBAL_SEED)
        gen_dataframe = pd.DataFrame(zip(gen_dataset.nonzero()[0], gen_dataset.nonzero()[1]))
        result_path = os.path.join(f'{RESULT_DIR}', f'{data.name}_p{change_prob}_eps{eps_}.tsv')
        gen_dataframe.to_csv(result_path, sep='\t', header=False, index=False)
        print(f'Dataset stored at: \'{result_path}\'')


def generate(data, change_probability, seed):
    ratings_generator = RandomizeResponse(change_probability=change_probability, base_seed=seed)
    return ratings_generator.privatize(data.values)


def exponential_mechanism(scores_path, eps):
    assert os.path.exists(scores_path), f'Scores not found at \'{scores_path}\''

    score_function = LoadScores(scores_path, sensitivity=1)
    computed_scores = list(score_function.data.keys())
    mech = ExponentialMechanism(score_function, eps)
    chosen_seed = mech.privatize(computed_scores)
    chosen_score = score_function.score_function(chosen_seed)
    return chosen_seed, chosen_score
