from collect_results import run

dataset_name = 'gift'
dataset_type = 'train'

for base_seed in range(10, 11, 1):
    args = {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'base_seed': base_seed
    }
    run(args)
