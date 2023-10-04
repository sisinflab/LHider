import multiprocessing as mp
import os.path
from src.jobs import experiment_info
from src.loader import *
from src.loader.paths import *
from src.dataset.dataset import *
from src.exponential_mechanism.scores import *
from src.randomize_response.mechanism import RandomizeResponse


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
    d_type = args['type']
    d_path = dataset_filepath(d_name, d_type)

    # privacy budget and change probability
    # probability of making a change in the dataset based on the privacy budget
    # prob = frac{1, (1 + e^{eps})}
    eps = float(args['eps'])
    change_prob = 1 / (1 + math.exp(eps))

    print(f'Change probability: {change_prob}')
    score_type = args['score_type']
    scores_folder = create_score_directory(d_name, eps, d_type, score_type)

    # loading files
    loader = TsvLoader(path=d_path, return_type="csr")
    dataset = DPCrsMatrix(loader.load(), path=d_name)

    # print dataset info
    dataset.info()

    # dataset in np.array for score computation
    data = np.array(dataset.dataset.todense())


    # generation parameters
    base_seed, start, end, batch, n_procs = args['base_seed'], args['start'], args['end'], args['batch'], args['proc']
    assert end >= start

    if n_procs > 1:
        run_batch_mp(data, change_prob, score_type, base_seed, start, end, batch, scores_folder, n_procs)
    else:
        run_batch(data, change_prob, score_type, base_seed, start, end, batch, scores_folder)


def compute_score(a: np.ndarray, b: np.ndarray, score_type: str) -> np.ndarray:
    scorer = SCORERS[score_type](a)
    return scorer.score_function(b)


def gen_and_score(data: np.ndarray, randomizer: RandomizeResponse, score_type: str, seed: int) -> [int, int]:
    generated_dataset = randomizer.privatize_np(input_data=data, relative_seed=seed)
    score = compute_score(data, generated_dataset, score_type)
    return score


def run_batch(data: np.ndarray, change_probability: float, score_type: str, base_seed: int,
              start: int, end: int, batch: int, result_dir: str):
    # check start and end
    assert end >= start
    assert batch > 0

    # compute batches indices
    batches = ((incremental_seed, min(incremental_seed + batch, base_seed + end))
               for incremental_seed in range(base_seed + start, base_seed + end, batch))

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

            score = gen_and_score(data=data, score_type=score_type, seed=data_seed, randomizer=randomizer)
            batch_results[data_seed] = score

        # store results
        batch_path = os.path.join(result_dir, f'batch_seed_{batch_start}_{batch_end}.pk')
        with open(batch_path, 'wb') as batch_file:
            pickle.dump(batch_results, batch_file)
        # keep track of batch files
        batch_paths.append(batch_path)

    aggregate_scores(score_paths=batch_paths, output_folder=result_dir, delete=True)


def run_batch_mp(data: np.ndarray, change_prob: float, score_type: str, seed: int,
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
    args = ((data, change_prob, score_type, seed, b[0], b[1], batch, path)
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


def scores_output_file(output_folder, scores):
    """
    Given the output folder and a collection of scores, returns the path of the file containing the scores
    @param output_folder: folder that will contain the scores
    @param output_folder: folder that will contain the scores
    @param scores: dictionary containing all the scores
    @return: a path
    """
    max_seed = max(scores.keys())
    min_seed = min(scores.keys())
    n = len(scores)
    output_path = os.path.join(output_folder, f'seed_{min_seed}_{max_seed}_n{n}.pk')
    return os.path.abspath(output_path)


def aggregate_scores(score_paths: list, output_folder: str, delete: bool = False) -> str:
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
        print(f'Output folder \'{output_folder}\' has been created')

    print('Reading score paths')
    aggregated_scores = {}
    for score_path in score_paths:
        with open(score_path, 'rb') as file:
            scores = pickle.load(file)
            aggregated_scores.update(scores)
            print(f'Scores loaded from \'{score_path}\'')

    print('Storing aggregation')
    output_path = scores_output_file(output_folder=output_folder, scores=aggregated_scores)

    with open(output_path, 'wb') as file:
        pickle.dump(aggregated_scores, file)
    print(f'Aggregated scores stored at \'{output_path}\'')

    # remove old score files
    if delete:
        for score_path in score_paths:
            os.remove(score_path)

    return output_path
