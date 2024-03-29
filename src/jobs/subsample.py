from . import experiment_info
from src.loader.paths import *
from src.loader import *
from src.dataset.dataset import *
from src.randomize_response import *
from src.lhider_mechanism import *
from src.exponential_mechanism import *
from src.laplace_mechanism.mechanism import *


def save_result(dataset, args, seed, eps_phi, total_eps, eps_exp):
    result = from_csr_to_pandas(csr_matrix(dataset))

    # STORE RESULT
    result_directory = noisy_dataset_folder(args['dataset'], args['type'], args['base_seed'])
    if not (os.path.exists(result_directory)):
        os.makedirs(result_directory)
        print(f'Directory created at \'{result_directory}\'')

    file_name = noisy_dataset_filename(args['randomizer'], eps_phi, args['reps'], eps_exp,
                                       seed, total_eps)
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


def run_subsampled(args: dict) -> None:
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

    sensitivity = SCORES_PROPERTIES[args['score_type']]['sensitivity']
    range = SCORES_PROPERTIES[args['score_type']]['range']

    RANDOMIZERS = {
        'randomized': RandomizeResponse,
        'discretized': DiscreteLaplaceMechanism
    }

    seed = args['seed']

    if args['reps'] == 1:
        for eps_exp in args['eps_exp']:
            seed += 1
            total_eps = round(args['eps_phi'] + (eps_exp / 2 * (1 + sensitivity / range)), 10)
            randomizer = RandomizeResponse(eps=0,
                                           sensitivity=1,
                                           base_seed=seed,
                                           min_val=0,
                                           max_val=5)
            mech = LHider(randomizer=randomizer,
                          n=args['reps'],
                          score=args['score_type'],
                          eps_exp=total_eps,
                          seed=seed)
            print("Genero " + str(args['reps']) + " dataset")
            randoms = mech.outputs(data)
            randomized = randoms[0]
            print(mech.file_name(dataset_name))
            save_result(randomized, args, seed, 0, total_eps, eps_exp)

    else:
        randomizer = RandomizeResponse(eps=0,
                                       sensitivity=1,
                                       base_seed=seed,
                                       min_val=0,
                                       max_val=5)
        mech = LHider(randomizer=randomizer,
                      n=args['reps'],
                      score=args['score_type'],
                      eps_exp=None,
                      seed=seed)

        print("Genero " + str(args['reps']) + " dataset")
        randoms = mech.outputs(data)

        exponential_mech = mech.exp_mech(randoms, data)
        print("Calcolo gli score")
        scores_ = exponential_mech.scores(randoms)

        for eps_exp in args['eps_exp']:
            seed += 1
            total_eps = round(args['eps_phi'] + (eps_exp / 2 * (1 + sensitivity / range)), 10)

            mech.set_exp_eps(total_eps, exponential_mech)
            randomized = exponential_mech.run_exponential(randoms, scores_)

            print(mech.file_name(dataset_name))
            save_result(randomized, args, seed, 0, total_eps, eps_exp)
