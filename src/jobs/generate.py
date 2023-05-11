import tqdm
from src.loader import *
from src.loader.paths import *
from src.recommender.neighbours import ItemKNNDense, ItemKNNSparse
from src.dataset.dataset import *
from src.exponential_mechanism.scores import MatrixCosineSimilarity
from src.randomize_response.mechanism import RandomizeResponse
import multiprocessing as mp
import os
import pickle
from src.jobs.aggregate import aggregate_scores


def experiment_info(arguments: dict):
    """
    Print information about the parameters of the experiments
    @param arguments: dictionary containing the parameters
    @return: None
    """
    print('Running generation job in dense configuration')
    for arg, value in arguments.items():
        print(f'{arg}: {value}')


def make_scores_folder(dataset_name, eps_rr):
    """
    Create the folder where the scores will be stored
    @param dataset_name: name of the dataset and name of the folder
    @return: path of the directory containing the scores
    """
    folder_path = os.path.abspath(os.path.join(DATA_DIR, dataset_name, 'scores', eps_rr))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'Scores directory has been created at \'{folder_path}\'')
    return folder_path


def dataset_path(dataset_name):
    """
    Given the name of the dataset return the path of the relative dataset file
    @param dataset_name: name of the dataset
    @return: path of the dataset
    """
    return os.path.abspath(os.path.join(DATA_DIR, dataset_name, 'dataset.tsv'))


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

    # paths
    d_name = args['dataset']
    d_path = dataset_path(d_name)

    # privacy budget and change probability
    eps = float(args['eps'])
    change_prob = 1 / (1 + math.exp(eps))

    print(f'Change probability: {change_prob}')
    scores_folder = make_scores_folder(d_name, f'eps_{eps}')

    # loading files
    loader = TsvLoader(path=d_path, return_type="csr")
    dataset = DPCrsMatrix(loader.load(), path=d_name)

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
    data = np.array(dataset.dataset.todense())
    ratings = compute_recommendations(data, model_name='itemknn')

    # generation parameters
    base_seed, start, end, batch, n_procs = args['base_seed'], args['start'], args['end'], args['batch'], args['proc']
    assert end >= start

    if n_procs > 1:
        run_batch_mp(data, ratings, change_prob, base_seed, start, end, batch, scores_folder, n_procs)
    else:
        run_batch(data, ratings, change_prob, base_seed, start, end, batch, scores_folder)


def compute_recommendations(data: np.ndarray, model_name: str) -> np.ndarray:
    """
    Compute the recommendation for a given model
    @param data: np.ndarray of the data
    @param model_name: string with the name of the model to compute the recommendation
    @return: np.ndarray containing the recommendation for the given data
    """
    MODEL_NAMES = ['itemknn']
    MODELS = {'itemknn': ItemKNNDense}
    assert model_name in MODEL_NAMES, f'Model missing. Model name {model_name}'
    model = MODELS[model_name](data, k=20)
    return model.fit()


def compute_score(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    scorer = MatrixCosineSimilarity(a)
    return scorer.score_function(b)


def gen_and_score(data: np.ndarray, randomizer: RandomizeResponse, ratings: np.ndarray, seed: int, difference: bool = True) -> [int, int]:
    generated_dataset = randomizer.privatize_np(input_data=data, relative_seed=seed)
    generated_ratings = compute_recommendations(generated_dataset, 'itemknn')
    score = compute_score(ratings, generated_ratings)

    if difference:
        diff = np.sum(data != generated_dataset)
        return {"score": score, "diff": diff}

    return score


def run_batch_mp(data: np.ndarray, ratings: np.ndarray, change_prob: float, seed: int,
                 start: int, end: int, batch: int, result_dir: str, n_procs: int):
    print('Running multiprocessing batch')
    print(f'Num processes: {n_procs}')
    print(f'Scores will be stored at \'{result_dir}\'')

    procs_batch = math.ceil((end - start) / n_procs)
    batch = min(procs_batch, batch)

    print(f'Batch size: {batch}')

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

    # aggregate all results
    score_paths = []
    for process_path in procs_path:
        for score in os.listdir(process_path):
            if '.pk' in score:
                score_paths.append(os.path.join(process_path, score))
    aggregate_scores(score_paths=score_paths, output_folder=result_dir)

    # delete old scores folders
    import shutil
    for process_path in procs_path:
        shutil.rmtree(process_path)
        print(f'Folder removed: \'{process_path}\'')


def run_batch(data: np.ndarray, ratings: np.ndarray, change_probability: float, base_seed: int,
              start: int, end: int, batch: int, result_dir: str):

    # check start and end
    assert end >= start
    assert batch > 0

    # compute batches indices
    batches = ((incremental_seed, min(incremental_seed + batch, base_seed + end))
               for incremental_seed in range(base_seed + start, base_seed + end, batch))

    # results accumulator
    aggregate_scores = {}

    # create randomizer class
    randomizer = RandomizeResponse(change_probability=change_probability, base_seed=base_seed)
    batch_paths = []

    # compute all the batches - batch start and batch end are absolute seeds
    for batch_start, batch_end in batches:
        # batch results accumulator
        batch_results = dict()
        # progress bar
        iterator = tqdm.tqdm(range(batch_start, batch_end))

        for data_seed in iterator:
            # progress bar update
            iterator.set_description(f'running seed {data_seed} in batch {batch_start} - {batch_end}')

            randomized_info = gen_and_score(data=data, ratings=ratings, seed=data_seed, randomizer=randomizer, difference=False)
            batch_results[data_seed] = randomized_info

        # update total score results
        aggregate_scores.update(batch_results)

        # store results
        batch_path = os.path.join(result_dir, f'batch_seed_{batch_start}_{batch_end}.pk')
        with open(batch_path, 'wb') as batch_file:
            pickle.dump(batch_results, batch_file)
        # keep track of batch files
        batch_paths.append(batch_path)

    # store final results
    print('Storing aggregation')
    max_seed = max(aggregate_scores.keys())
    min_seed = min(aggregate_scores.keys())
    n = len(aggregate_scores)

    output_path = os.path.join(result_dir, f'seed_{min_seed}_{max_seed}_n{n}.pk')
    with open(output_path, 'wb') as batch_file:
        pickle.dump(aggregate_scores, batch_file)

    # remove temporary batch results files
    for bp in batch_paths:
        os.remove(bp)
        print(f'File removed: \'{bp}\'')
