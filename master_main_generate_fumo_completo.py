from src.jobs.sigir_itemknn import run_generation

# definisci i parametri necessari
randomizer = 'randomized'
exp_score = 'jaccard'

# dataset
dataset_name = 'yahoo_movies'
dataset_type = 'train'
n = 170
folder = 2
seed = 341

def fun():
    for eph_phi in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
        # run
        args = {
            'dataset': dataset_name,
            'type': dataset_type,
            'eps_phi': eph_phi,
            'randomizer': randomizer,
            'base_seed': folder,
            'score_type': exp_score,
            'generations': n,
            'seed': seed
        }
        run_generation(args)

from email_notifier.email_sender import EmailNotifier
notifier = EmailNotifier()
arguments = {'Esperimento': 'ultima speranza yahoo 341 ...'}
notifier.notify(fun, additional_body=str(arguments))
