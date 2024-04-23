from . import experiment_info
from src.loader.paths import *
from src.loader import *
from src.dataset.dataset import *
from src.randomize_response import *
from src.lhider_mechanism import *
from src.exponential_mechanism import *
from src.laplace_mechanism.mechanism import *


def run(args: dict):
    # print information about the experiment
    experiment_info(args)

    # check for the existence of fundamental directories, otherwise create them
    check_main_directories()

    # dataset directory
    dataset_name = args['dataset']
    dataset_type = args['type']

    # loading files
    dataset_path = dataset_filepath(dataset_name, dataset_type)
    loader = TsvLoader(path=dataset_path, return_type="sparse")
    data = loader.load().todense()

    RANDOMIZERS = {
        'randomized': RandomizeResponse,
        'discretized': DiscreteLaplaceMechanism
    }

    # randomizer = RANDOMIZERS[args['randomizer']](epsilon=args['eps_phi'], base_seed=args['seed'])
    randomizer = RandomizeResponse(eps=args['eps_phi'],
                                   sensitivity=1,
                                   base_seed=args['seed'],
                                   min_val=0,
                                   max_val=5)

    for eps_exp in args['eps_exp']:
        total_eps = args['eps_phi'] + eps_exp

        mech = LHider(randomizer=randomizer,
                      n=args['reps'],
                      score='manhattan',
                      eps_exp='eps_exp',
                      seed=args['seed'])

        randomized = mech.privatize_range(data).reshape(data.shape)
        print(mech.file_name(dataset_name))
        result = from_csr_to_pandas(csr_matrix(randomized))

        # STORE RESULT
        result_directory = noisy_dataset_folder(dataset_name, args['type'], args['base_seed'])
        if not (os.path.exists(result_directory)):
            os.makedirs(result_directory)
            print(f'Directory created at \'{result_directory}\'')

        file_name = noisy_dataset_filename(args['randomizer'], args['eps_phi'], args['reps'], args['eps_exp'],
                                           args['seed'], total_eps)
        file_path = os.path.join(result_directory, file_name + '.tsv')
        result.to_csv(file_path, sep='\t', header=False, index=False)
        print(f'File stored at \'{file_path}\'')


def run_explicit(args: dict):
    # print information about the experiment
    experiment_info(args)

    # check for the existence of fundamental directories, otherwise create them
    check_main_directories()

    # dataset directory
    dataset_name = args['dataset']
    dataset_type = args['type']

    # loading files
    dataset_path = dataset_filepath(dataset_name, dataset_type)
    loader = TsvLoader(path=dataset_path, return_type="sparse")
    data = loader.load().todense()

    print()

    RANDOMIZERS = {
        'randomized': RandomizeResponse,
        'discretized': DiscreteLaplaceMechanism
    }

    # randomizer = RANDOMIZERS[args['randomizer']](epsilon=args['eps_phi'], base_seed=args['seed'])
    randomizer = DiscreteLaplaceMechanism(eps=args['eps_phi'],
                                          sensitivity=1,
                                          base_seed=args['seed'],
                                          min_val=0,
                                          max_val=5)

    mech = LHider(randomizer=randomizer,
                  n=args['reps'],
                  score='manhattan',
                  eps_exp=args['eps_exp'],
                  seed=args['seed'])

    randomized_data = mech.privatize_matrix(data).reshape(data.shape)
    print(mech.file_name(dataset_name))
    result = from_csr_to_pandas(csr_matrix(randomized_data), explicit=True)

    # STORE RESULT
    result_directory = noisy_dataset_folder(dataset_name, args['type'], args['base_seed'])
    if not (os.path.exists(result_directory)):
        os.makedirs(result_directory)
        print(f'Directory created at \'{result_directory}\'')

    file_name = noisy_dataset_filename(args['randomizer'], args['eps_phi'], args['reps'], args['eps_exp'], args['seed'],
                                       args['total_eps'])
    file_path = os.path.join(result_directory, file_name + '.tsv')
    result.to_csv(file_path, sep='\t', header=False, index=False)
    print(f'File stored at \'{file_path}\'')


def run_new_expo(args: dict):
    # print information about the experiment
    experiment_info(args)

    # check for the existence of fundamental directories, otherwise create them
    check_main_directories()

    # dataset directory
    dataset_name = args['dataset']
    dataset_type = args['type']

    # loading files
    dataset_path = dataset_filepath(dataset_name, dataset_type)
    loader = TsvLoader(path=dataset_path, return_type="sparse")
    data = loader.load().A

    SCORES_PROPERTIES = {
        'manhattan':
            {
                'sensitivity': lambda x: 1/sum(sum(x)),
                'range': 1
            },
        'jaccard':
            {
                'sensitivity': 0,
                'range': 1
            }
    }
    sensitivity = SCORES_PROPERTIES[args['score_type']]['sensitivity'](data)
    range = SCORES_PROPERTIES[args['score_type']]['range']
    print()

    RANDOMIZERS = {
        'randomized': RandomizeResponse,
        'discretized': DiscreteLaplaceMechanism
    }

    # randomizer = RANDOMIZERS[args['randomizer']](epsilon=args['eps_phi'], base_seed=args['seed'])
    if args['reps'] == 1:
        for eps_exp in args['eps_exp']:
            total_eps = round(args['eps_phi'] + (eps_exp / 2 * (1 + sensitivity / range)), 10)
            randomizer = RandomizeResponse(eps=total_eps,
                                           sensitivity=1,
                                           base_seed=args['seed'],
                                           min_val=0,
                                           max_val=5)
            mech = LHider(randomizer=randomizer,
                          n=args['reps'],
                          score=args['score_type'],
                          eps_exp=eps_exp,
                          seed=args['seed'])
            print("Genero " + str(args['reps']) + " dataset")
            randoms = mech.outputs(data)
            randomized = randoms[0]
            print(mech.file_name(dataset_name))
            save_result(randomized, args, total_eps, eps_exp)
    else:
        randomizer = RandomizeResponse(eps=args['eps_phi'],
                                       sensitivity=1,
                                       base_seed=args['seed'],
                                       min_val=0,
                                       max_val=5)
        mech = LHider(randomizer=randomizer,
                      n=args['reps'],
                      score=args['score_type'],
                      eps_exp=None,
                      seed=args['seed'])

        print("Genero " + str(args['reps']) + " dataset")
        randoms = mech.outputs(data)

        exponential_mech = mech.exp_mech(randoms, data)
        print("Calcolo gli score")
        scores_ = exponential_mech.scores(randoms)

        for eps_exp in args['eps_exp']:
            total_eps = round(args['eps_phi'] + (eps_exp / 2 * (1 + sensitivity / range)), 10)

            mech.set_exp_eps(eps_exp, exponential_mech)
            randomized = exponential_mech.run_exponential(randoms, scores_)

            print(mech.file_name(dataset_name))
            save_result(randomized, args, total_eps, eps_exp)


def run_generation(args: dict):
    # print information about the experiment
    experiment_info(args)

    # dataset directory
    dataset_name = args['dataset']
    dataset_type = args['type']
    return_type = args.get('return_type', 'sparse')

    # loading files
    dataset_path = dataset_filepath(dataset_name, dataset_type)
    loader = TsvLoader(path=dataset_path, return_type=return_type)
    data = loader.load().A

    if args['randomizer'] == 'randomized':
        randomizer = RandomizeResponse(eps=args['eps_phi'],
                                       sensitivity=1,
                                       base_seed=0,
                                       min_val=0,
                                       max_val=5)
    elif args['randomizer'] == 'subsampled':
        randomizer = RandomGenerator()
    elif args['randomizer'] == 'discrete_laplace':
        randomizer = DiscreteLaplaceMechanism(eps=args['eps_phi'], sensitivity=args['max_val'] - args['min_val'], min_val=args['min_val'], max_val=args['max_val'])
    else:
        print('randomizer not implemented')

    score_type = args['score_type']
    for actual_seed in range(args['seed'], args['seed'] + args['generations']):
        print(f'Generating a new random dataset {dataset_name} {dataset_type} with seed \'{actual_seed}\'')
        output = randomizer.privatize_np(data, relative_seed=actual_seed)
        assert score_type in SCORES, f'Score type not found. Accepted scores: {SCORES.keys()}'
        scorer = SCORES[score_type](data)
        score = scorer.score_function(output)
        save_generated(output, actual_seed, score, args)


def save_generated(data, actual_seed, score, args):
    if args.get('ml_task'):
        result = pd.DataFrame(data)
    else:
        result = from_csr_to_pandas(csr_matrix(data))
    result_directory = noisy_dataset_folder(args['dataset'], args['type'], args['base_seed'])
    if not (os.path.exists(result_directory)):
        os.makedirs(result_directory)
        print(f'Directory created at \'{result_directory}\'')
    file_name = noisy_dataset_filename(args['randomizer'], args['eps_phi'], 1, score, actual_seed, args['eps_phi'])
    file_path = os.path.join(result_directory, file_name + '.tsv')
    result.to_csv(file_path, sep='\t', header=False, index=False)
    print(f'File stored at \'{file_path}\'')
    return file_path

def save_result(dataset, args, total_eps, eps_exp):
    result = from_csr_to_pandas(csr_matrix(dataset))

    # STORE RESULT
    result_directory = noisy_dataset_folder(args['dataset'], args['type'], args['base_seed'])
    if not (os.path.exists(result_directory)):
        os.makedirs(result_directory)
        print(f'Directory created at \'{result_directory}\'')

    file_name = noisy_dataset_filename(args['randomizer'], args['eps_phi'], args['reps'], eps_exp,
                                       args['seed'], total_eps)
    file_path = os.path.join(result_directory, file_name + '.tsv')
    result.to_csv(file_path, sep='\t', header=False, index=False)
    print(f'File stored at \'{file_path}\'')


def from_csr_to_pandas(data: csr_matrix, explicit=False):
    u, i = data.nonzero()
    if explicit:
        r = data.data
        return pd.DataFrame(np.array([u, i, r]).T)
    return pd.DataFrame(np.array([u, i]).T)


def noisy_dataset_filename(randomizer, eps_phi, reps, eps_exp, seed, total_eps):
    return '_'.join([str(randomizer), str(eps_phi), str(reps), str(eps_exp),
                     str(seed), str(total_eps)])



def noisy_dataset_folder(dataset_name, dataset_type, base_seed):
    return os.path.join('perturbed_datasets', dataset_name + '_' + dataset_type, str(base_seed))