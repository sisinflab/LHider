from recs import run as run_recs

dataset_name = 'yahoo_movies'
dataset_type = 'train'

for base_seed in range(2200, 3601, 100):
    args = {
        'dataset_name': dataset_name,
        'dataset_type': dataset_type,
        'base_seed': base_seed
    }
    run_recs(args)
