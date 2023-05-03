import math

import tqdm

from src.loader import *
from src.loader.paths import *

from src.dataset.dataset import DPCrsMatrix
from scipy.sparse import csr_matrix
from src.exponential_mechanism.mechanism import ExponentialMechanism
from src.exponential_mechanism.scores import MatrixCosineSimilarity, LoadScores
from src.randomize_response.mechanism import RandomizeResponse
from src.recommender import ItemKNN
import pandas as pd
import multiprocessing as mp
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

    # job
    job = args['job']
    jobs = {'identity', 'gen_n_score', 'run_batch', 'run_batch_mp', 'randomize'}
    assert job in jobs

    if job == 'identity':
        print('running job IDENTITY')
        # recommendations
        print(f'\nComputing recommendations')
        ratings = compute_recommendations(data, model_name='itemknn')
        score = identity_score(ratings)
        print(f'identity score: {score}')
    elif job == 'gen_n_score':
        # recommendations
        print(f'\nComputing recommendations')
        ratings = compute_recommendations(data, model_name='itemknn')
        seed = args['seed']
        change_prob = args['change_prob']
        score, _ = gen_and_score(data, ratings, change_prob, seed=seed)
        print(score)
    elif job == 'run_batch':
        # recommendations
        print(f'\nComputing recommendations')
        ratings = compute_recommendations(data, model_name='itemknn')
        seed = args['seed']
        change_prob = args['change_prob']
        start = args['start']
        end = args['end']
        batch = args['batch']
        run_batch(data, ratings, change_prob, seed, start, end, batch, dataset_result_dir)
    elif job == 'run_batch_mp':
        # recommendations
        print(f'\nComputing recommendations')
        ratings = compute_recommendations(data, model_name='itemknn')
        change_prob = args['change_prob']
        seed = args['seed']
        start = args['start']
        end = args['end']
        assert end >= start
        batch = args['batch']
        n_procs = args['proc']
        run_batch_mp(data, ratings, change_prob, seed, start, end, batch, dataset_result_dir, n_procs)
    elif job == 'randomize':
        path = args['final_path']
        eps = args['exp_eps']
        change_prob = args['change_prob']
        for eps_ in eps:
            c_seed, c_score = exponential_mechanism(path=path, eps=eps_)
            print(f'Selected a dataset with score: {c_score}')
            gen_dataset = generate(data=data, change_probability=change_prob, seed=c_seed-GLOBAL_SEED)
            gen_dataframe = pd.DataFrame(zip(gen_dataset.nonzero()[0], gen_dataset.nonzero()[1]))
            result_path = os.path.join(f'{RESULT_DIR}', f'{data.name}_eps{eps_}.tsv')
            gen_dataframe.to_csv(result_path, sep='\t', header=False, index=False)
            print(f'Dataset stored at: \'{result_path}\'')


def compute_recommendations(data, model_name):
    MODEL_NAMES = ['itemknn']
    MODELS = {'itemknn': ItemKNN}
    assert model_name in MODEL_NAMES, f'Model missing. Model name {model_name}'
    model = MODELS[model_name](data, k=20)
    return model.fit()


def generate(data, change_probability, seed):
    ratings_generator = RandomizeResponse(change_probability=change_probability, base_seed=seed)
    return ratings_generator.privatize(data.values)


def compute_score(a, b):
    scorer = MatrixCosineSimilarity(a)
    return scorer.score_function(b)


def fake(data, ratings, n_ratings, seed=0):
    return 0, seed


def gen_and_score(data, ratings, change_probability, seed=0):
    ratings_generator = RandomizeResponse(change_probability=change_probability, base_seed=seed)
    # TODO: usare privatize_np  se possibile
    generated_dataset = ratings_generator.privatize(data.values)
    generated_dataset = DPCrsMatrix(generated_dataset)
    generated_ratings = compute_recommendations(generated_dataset, 'itemknn')
    score = compute_score(ratings, generated_ratings)
    return score, seed


def gen_score_and_store(data, ratings, change_probability, seed=42):
    score, _ = gen_and_score(data=data,ratings=ratings, change_probability=change_probability, seed=seed)
    with open(os.path.join(RESULT_DIR, f'{seed}.pk'), 'wb') as file:
        pickle.dump(score, file)


def identity_score(ratings):
    return compute_score(ratings, ratings)


def run_mp(args):
    dataset_path = args['dataset']
    loader = TsvLoader(path=dataset_path, return_type=csr_matrix)
    data = DPCrsMatrix(loader.load())

    with mp.Pool(16) as pool:
        args = ((data, x) for x in range(100))
        results = pool.starmap_async(gen_score_and_store, args)
        results.get()


def run_batch(data, ratings, change_prob, seed, start, end, batch, result_dir):
    assert end >= start
    assert batch > 0

    # compute batches indices
    batches = ((incremental_seed, min(incremental_seed + batch, seed + end))
               for incremental_seed in range(seed + start, seed + end, batch))
    total_results = {}

    # compute all the batches
    for batch_start, batch_end in batches:
        batch_results = dict()
        iterator = tqdm.tqdm(range(batch_start, batch_end))
        for new_seed in iterator:
            iterator.set_description(f'running seed {new_seed} in batch {batch_start} - {batch_end}')
            score, gen_seed = gen_and_score(
                data=data, ratings=ratings, change_probability=change_prob, seed=new_seed)
            assert gen_seed == new_seed
            batch_results[new_seed] = score

        # update total score results
        total_results.update(batch_results)

        # store results
        batch_path = os.path.join(result_dir, f'seed_score_{batch_start}_{batch_end}.pk')
        with open(batch_path, 'wb') as batch_file:
            pickle.dump(batch_results, batch_file)

    # store final results
    result_path = os.path.join(result_dir, f'final_seed_{start+seed}_{end+seed}.pk')
    with open(result_path, 'wb') as batch_file:
        pickle.dump(total_results, batch_file)


def run_batch_mp(data, ratings, change_prob, seed, start, end, batch, result_dir, n_procs):
    print('running multiprocessing batch')
    print(f'n processes: {n_procs}')

    procs_batch = math.ceil((end - start) / n_procs)
    batch = min(procs_batch, batch)

    print(f'batch size: {batch}')

    procs_batches = ((idx, min(end, idx + procs_batch)) for idx in range(start, end, procs_batch))

    procs_path = [os.path.join(result_dir, str(pid)) for pid in range(n_procs)]
    for path in procs_path:
        if not os.path.exists(path):
            os.makedirs(path)
    args = ((data, ratings, change_prob, seed, b[0], b[1], batch, path)
            for b, path in zip(procs_batches, procs_path))

    with mp.Pool(n_procs) as pool:
        results = pool.starmap_async(run_batch, args)
        results.get()

def exponential_mechanism(path, eps):
    result_path = os.path.join('results', 'final', path)
    assert os.path.exists(result_path)

    score_function = LoadScores(result_path, sensitivity=1)
    computed_scores = list(score_function.data.keys())
    mech = ExponentialMechanism(score_function, eps)
    chosen_seed = mech.privatize(computed_scores)
    chosen_score = score_function.score_function(chosen_seed)
    return chosen_seed, chosen_score
