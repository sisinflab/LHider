from src.loader.paths import *
from src.loader.loaders import *
from src.dataset.dataset import *
from src.jobs.sigir import from_csr_to_pandas
from data_preprocessing.filters.filter import store_dataset


def run(args):
    dataset_name = args['dataset_name']
    dataset_type = args['dataset_type']
    dataset_path = dataset_filepath(dataset_name, dataset_type)

    loader = TsvLoader(path=dataset_path, return_type="csr")
    data = DPCrsMatrix(loader.load(), path=dataset_path, data_name=dataset_name)
    dataframe = from_csr_to_pandas(csr_matrix(data.dataset))
    dataframe['r'] = data.dataset.data

    dp = store_dataset(data=dataframe,
                       folder=os.path.join(dataset_directory(dataset_name), 'data'),
                       name='dataset_r',
                       message='Re-indexed dataset')['dataset_r']

    exit()

    loader2 = TsvLoader(path=dp, return_type='sparse')
    data = DPCrsMatrix(loader2.load(), path=dp, data_name='dataset_r')
    u, i = data.dataset.nonzero()
    r = data.dataset.data
    dataframe = pd.DataFrame(np.array([u, i, r]).T)
    print()
    dp = store_dataset(data=data,
                       folder=os.path.join(dataset_directory(dataset_name), 'data'),
                       name='dataset_r',
                       message='Re-indexed dataset')['dataset_r']
