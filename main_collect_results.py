from collect_results import run

dataset_name = 'yahoo_movies'
dataset_type = 'train'

for base_seed in range(200, 801, 100):
    args = {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'base_seed': base_seed
    }
    run(args)
