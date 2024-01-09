import os

from . import experiment_info
from src.loader.paths import *
from src.loader import *
from src.dataset.dataset import *
from src.randomize_response import *
from src.lhider_mechanism import *
from src.exponential_mechanism import *

def run(args: dict, **kwargs):

    # print information about the experiment
    experiment_info(args)

    # check for the existence of fundamental directories, otherwise create them
    check_main_directories()

    # dataset directory
    dataset_name = args['dataset']
    dataset_type = args['type']

    # loading files
    dataset_path = dataset_filepath(dataset_name, dataset_type)
    loader = TsvLoader(path=dataset_path, return_type="csr")
    data = DPCrsMatrix(loader.load(), path=dataset_path, data_name=dataset_name)
    print()

    RANDOMIZERS = {
        'randomized': RandomizeResponse
    }

    randomizer = RANDOMIZERS[args['randomizer']](epsilon=args['eps_phi'])
    data[0].todense()
    mech = LHider(randomizer=randomizer,
                  n=args['reps'],
                  score='manhattan',
                  eps_exp=args['eps_exp'],
                  seed=args['seed'])

    d = np.array(data.to_dense())
    r = mech.privatize_matrix(d).reshape(d.shape)
    print(mech.file_name(dataset_name))
    print()
    result = from_csr_to_pandas(csr_matrix(r))
    print()

    # STORE RESULT
    result_directory = os.path.join(RESULT_DIR, 'perturbed_datasets', dataset_name + '_' + args['type'])
    if not(os.path.exists(result_directory)):
        os.makedirs(result_directory)
        print(f'Directory created at \'{result_directory}\'')

    file_name = '_'.join([str(args['randomizer']), str(args['eps_phi']), str(args['reps']), str(args['eps_exp'])])
    file_path = os.path.join(result_directory, file_name + '.tsv')
    result.to_csv(file_path, sep='\t', header=False, index=False)
    print(f'File stored at \'{file_path}\'')

    print()



def from_csr_to_pandas(data: csr_matrix):
    u, i = data.nonzero()
    return pd.DataFrame(np.array([u, i]).T)
