import tqdm
from src.loader import *
from src.loader.paths import *
from src.dataset.dataset import *
from scipy.sparse import csr_matrix
from src.exponential_mechanism.scores import MatrixCosineSimilarity, LoadScores
from src.randomize_response.mechanism import RandomizeResponse
from src.recommender import ItemKNNSparse
import multiprocessing as mp
import os
import pickle


def experiment_info(arguments: dict):
    """
    Print information about the parameters of the experiments
    @param arguments: dictionary containing the parameters
    @return: None
    """
    print('Running generation job in sparse configuration')
    for arg, value in arguments.items():
        print(f'{arg}: {value}')


def run(args: dict):
    """
    Run the generate dataset job
    @param args: dictionary containing the parameters
    @return: None
    """
    # print information about the experiment
    experiment_info(args)

    # check the fundamental directories
    check_main_directories()

    # loading files
    dataset_path = args['dataset']
    loader = TsvLoader(path=dataset_path, return_type="csr")
    dataset = DPCrsMatrix(loader.load(), path=dataset_path)

    # dataset result directory
    dataset_result_dir = os.path.join(RESULT_DIR, dataset.name)
    create_directory(dataset_result_dir)

    # print dataset info
    # TODO: implementare qui il metodo info della classe dataset
    print(f'data ratings: {dataset.transactions}')
    print(f'data users: {dataset.n_users}')
    print(f'data items: {dataset.n_items}')

    # recommendations returned as a np.array
    print(f'\nComputing recommendations')
    ratings = compute_recommendations(dataset, model_name='itemknn')

    change_prob, base_seed, start, end, batch, n_procs = \
        args['change_prob'], args['base_seed'], args['start'], args['end'], args['batch'], args['proc']
    assert end >= start

    if n_procs > 1:
        run_batch_mp(dataset, ratings, change_prob, base_seed, start, end, batch, dataset_result_dir, n_procs)
    else:
        run_batch(dataset, ratings, change_prob, base_seed, start, end, batch, dataset_result_dir)


def compute_recommendations(data, model_name: str):
    """
    Compute the recommendation for a given model
    @param data: np.ndarray of the data
    @param model_name: string with the name of the model to compute the recommendation
    @return: np.ndarray containing the recommendation for the given data
    """
    MODEL_NAMES = ['itemknn']
    MODELS = {'itemknn': ItemKNNSparse}
    assert model_name in MODEL_NAMES, f'Model missing. Model name {model_name}'
    model = MODELS[model_name](data, k=20)
    return model.fit()


def compute_score(a, b):
    scorer = MatrixCosineSimilarity(a)
    return scorer.score_function(b)


def gen_and_score(data, randomizer: RandomizeResponse, ratings, seed: int, difference: bool = True) -> [int, int]:
    generated_dataset = randomizer.privatize(input_data=data.dataset, relative_seed=seed)
    generated_dataset = DPCrsMatrix(generated_dataset)
    generated_ratings = compute_recommendations(generated_dataset, 'itemknn')
    score = compute_score(ratings, generated_ratings)
    if difference:
        diff = csr_matrix.sum(data.dataset != generated_dataset.dataset)
        return {"score": score, "diff": diff}

    return score


def run_batch_mp(data: np.ndarray, ratings: np.ndarray, change_prob: float, seed: int,
                 start: int, end: int, batch: int, result_dir: str, n_procs: int):
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


def run_batch(data: np.ndarray, ratings: np.ndarray, change_probability: float, base_seed: int,
              start: int, end: int, batch: int, result_dir: str):

    # check start and end
    assert end >= start
    assert batch > 0

    # compute batches indices
    batches = ((incremental_seed, min(incremental_seed + batch, base_seed + end))
               for incremental_seed in range(base_seed + start, base_seed + end, batch))

    # results accumulator
    total_results = {}

    # create randomizer class
    randomizer = RandomizeResponse(change_probability=change_probability, base_seed=base_seed)

    # compute all the batches - batch start and batch end are absolute seeds
    for batch_start, batch_end in batches:
        # batch results accumulator
        batch_results = dict()
        # progress bar
        iterator = tqdm.tqdm(range(batch_start, batch_end))

        for data_seed in iterator:
            # progress bar update
            iterator.set_description(f'running seed {data_seed} in batch {batch_start} - {batch_end}')

            randomized_info = gen_and_score(data=data, ratings=ratings, seed=data_seed, randomizer=randomizer, difference=True)
            batch_results[data_seed] = randomized_info

        # update total score results
        total_results.update(batch_results)

        # store results
        batch_path = os.path.join(result_dir, f'batch_seed_{batch_start}_{batch_end}.pk')
        with open(batch_path, 'wb') as batch_file:
            pickle.dump(batch_results, batch_file)

    # store final results
    result_path = os.path.join(result_dir, f'seed_{start + base_seed}_{end + base_seed}.pk')
    with open(result_path, 'wb') as batch_file:
        pickle.dump(total_results, batch_file)
