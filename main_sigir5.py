from src.jobs.sigir import run, run_explicit, run_new_expo
from composition import tight_adv_comp


# definisci i parametri necessari
randomizer = 'randomized'
exp_score = 'manhattan'

# dataset
dataset_name = 'yahoo_movies'
dataset_type = 'train'

from email_notifier.email_sender import EmailNotifier
notifier = EmailNotifier()
arguments = {'Esperimento': 'generazione'}


def fun(*args, **kwargs):
    for base_seed in range(100, 1001, 100):
        seed = base_seed
        for eph_phi in [1, 2, 3, 5, 10, 15]:
            for eps_exp in [0.001, 0.005, 0.01, 0.05]:
                seed += 1
                reps = 1

                # items = 1034
                # delta = (1 / items) ** 1.1
                # new_eps = tight_adv_comp(desired_gens, desired_eps, delta)
                total_exp = eps_exp + eph_phi
                # run
                args = {
                    'dataset': dataset_name,
                    'type': dataset_type,
                    'eps_phi': total_exp,
                    'randomizer': randomizer,
                    'reps': reps,
                    'eps_exp': eps_exp,
                    'seed': seed,
                    'base_seed': base_seed,
                    'total_eps': total_exp,
                    'score_type': 'manhattan'
                }
                run_new_expo(args)

notifier.notify(fun, {}, additional_body=str(arguments))
