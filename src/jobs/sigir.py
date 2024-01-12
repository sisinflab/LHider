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
    data = DPCrsMatrix(loader.load(), path=dataset_path, data_name=dataset_name)

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
    data[0].todense()

    mech = LHider(randomizer=randomizer,
                  n=args['reps'],
                  score='manhattan',
                  eps_exp=args['eps_exp'],
                  seed=args['seed'])

    d = np.array(data.to_dense())
    r = mech.privatize_matrix(d).reshape(d.shape)
    print(mech.file_name(dataset_name))
    result = from_csr_to_pandas(csr_matrix(r))

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
