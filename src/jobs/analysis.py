from src.loader import *
from src.loader.paths import *
from src.dataset.dataset import DPCrsMatrix
from src.exponential_mechanism.mechanism import ExponentialMechanism
from src.exponential_mechanism.scores import *
from src.randomize_response.mechanism import RandomizeResponse
import os, statistics
from matplotlib import pyplot as plt

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
    scores = Scores(path=scores_path)
    scores.load()

    # privacy parameters
    change_prob = args['change_prob']

    # score_range(scores)
    # score_frequency(scores)
    # score_variance(scores)
    # score_max(scores)

    id_to_diff(data, scores, change_prob)


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


def score_range(scores):
    print(f'max score: {max(scores.data.values())}')
    print(f'min score: {min(scores.data.values())}')


def score_variance(scores, plot=True):
    data = scores.to_dataframe()

    variance = {}
    for i in range(100, len(data), 100):
        variance[i] = data[:i].scores.var()

    plt.plot(variance.keys(), variance.values())
    plt.show()
    print()

def score_max(scores, plot=True):
    data = scores.to_dataframe()

    max_vals = {}
    for i in range(100, len(data), 100):
        max_vals[i] = data[:i].scores.max()

    plt.plot(max_vals.keys(), max_vals.values())
    plt.show()
    print()


def score_frequency(scores, decimals=5, plot=True):

    scores.decimal(decimals=decimals)
    data = scores.to_dataframe()
    count = data.scores.value_counts()
    count = count.sort_index()
    if plot:
        plt.plot(count)
        plt.show()


def id_to_diff(data, scores, change_probability):
    # impostiamo il base seed a 0 perché abbiamo il valore assoluto del seed
    ratings_generator = RandomizeResponse(change_probability=change_probability, base_seed=0)
    differences = {}
    data = data.values.todense()
    for seed, score in tqdm.tqdm(scores.data.items()):
        generated = ratings_generator.privatize_np(data, seed)
        diff = 0
        #diff = scipy.sparse.csr_matrix.sum(data.dataset != generated)
        differences[diff] = score

    print()
    return ratings_generator.privatize(data.values)
