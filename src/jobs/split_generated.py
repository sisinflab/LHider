from src.loader.paths import *
from data_preprocessing.filters.dataset import Splitter
from data_preprocessing.filters.filter import store_dataset
from src.dataset.dataset import *
from src.jobs.sigir import noisy_dataset_folder


def run(args):
    dataset_name = args['dataset_name']
    dataset_type = args['dataset_type']

    base_seed = args['base_seed']

    result_dir = noisy_dataset_folder(dataset_name, dataset_type, base_seed)
    files = os.listdir(result_dir)

    generated_basefolder = os.path.join(dataset_directory(dataset_name), 'generated_' + dataset_type)
    if os.path.exists(generated_basefolder) is False:
        os.makedirs(generated_basefolder)
        print(f'created directory \'{generated_basefolder}\'')

    generated_folder = os.path.join(generated_basefolder, str(base_seed))
    if os.path.exists(generated_folder) is False:
        os.makedirs(generated_folder)
        print(f'created directory \'{generated_folder}\'')

    for file in files:
        if '.tsv' in file:
            data_name = file.replace('.tsv', '')
            dataset_path = os.path.join(result_dir, file)

            dataset = pd.read_csv(dataset_path, sep='\t', header=None, names=['u', 'i', 'r'])
            print(f'dataset loaded from \'{dataset_path}\'')

            splitter = Splitter(data=dataset,
                                test_ratio=0.2)
            try:
                splitting_results = splitter.filter()

                data_folder = os.path.join(generated_folder, data_name)
                if os.path.exists(data_folder) is False:
                    os.makedirs(data_folder)
                    print(f'created directory \'{data_folder}\'')

                train = splitting_results["train"]
                val = splitting_results["test"]

                train['r'] = 1
                val['r'] = 1

                store_dataset(data=splitting_results["train"],
                              folder=data_folder,
                              name='train',
                              message='training set')

                store_dataset(data=splitting_results["test"],
                              folder=data_folder,
                              name='validation',
                              message='val set')
            except Exception as e:
                print(e)
                print(f'DATASET NON SPLITTATO {dataset_name}')

