from src.loader.paths import *
from elliot.run import run_experiment
from config_templates.training import TEMPLATE_PATH
import tqdm

DEFAULT_METRICS = ["nDCGRendle2020", "Recall", "HR", "nDCG", "Precision", "F1", "MAP", "MAR", "ItemCoverage", "Gini",
                   "SEntropy", "EFD", "EPC", "PopREO", "PopRSP", "ACLT", "APLT", "ARP"]


def run(args):
    dataset_name = args['dataset_name']
    dataset_type = args['dataset_type']
    base_seed = args['base_seed']
    from src.jobs.sigir import noisy_dataset_folder

    result_dir = noisy_dataset_folder(dataset_name=dataset_name,
                                      dataset_type=dataset_type,
                                      base_seed=base_seed)

    files = os.listdir(result_dir)

    test_path = dataset_filepath(dataset_name, 'test')

    output_folder = os.path.join(PROJECT_PATH, 'results_collection', dataset_name + '_' + dataset_type, str(base_seed))
    if os.path.exists(output_folder) is False:
        os.makedirs(output_folder)
        print(f'Created folder at \'{output_folder}\'')
    print(f'Results will be stored at \'{output_folder}\'')

    for file in tqdm.tqdm(files):
        if '.tsv' in file:
            data_name = file.replace('.tsv', '')

            data_folder = os.path.join(dataset_directory(dataset_name),
                                       'generated_' + dataset_type, str(base_seed),
                                       data_name)
            train_path = os.path.join(data_folder, 'train.tsv')
            val_path = os.path.join(data_folder, 'validation.tsv')

            try:
                assert os.path.exists(train_path)
                assert os.path.exists(val_path)
                assert os.path.exists(test_path)

                result_folder = os.path.join(output_folder, data_name)
                os.makedirs(result_folder)

                config = TEMPLATE_PATH.format(dataset=data_name,
                                              output_path=result_folder,
                                              train_path=train_path,
                                              val_path=val_path,
                                              test_path=test_path)

                config_path = os.path.join(CONFIG_DIR, 'runtime.yml')
                with open(config_path, 'w') as conf_file:
                    conf_file.write(config)

                run_experiment(config_path)
            except Exception as e:
                print(e)
                print(f'Keep attention: recommendation has not been computer for {dataset_name}')
