from src.jobs.sigir import run_new_expo

# definisci i parametri necessari
randomizer = 'randomized'
exp_score = 'jaccard'

# dataset
dataset_name = 'yahoo_movies'
dataset_type = 'train'


def fun():
    for base_seed in range(400, 701, 100):
        seed = base_seed
        for eph_phi in [1, 2, 3, 5, 10, 15]:
            for reps in [10, 100, 1000]:
                for eps_exp in [0.001, 0.005, 0.01, 0.05]:
                    seed += 1

                    total_eps = eph_phi + eps_exp
                    # run
                    args = {
                        'dataset': dataset_name,
                        'type': dataset_type,
                        'eps_phi': eph_phi,
                        'randomizer': randomizer,
                        'reps': reps,
                        'eps_exp': eps_exp,
                        'seed': seed,
                        'base_seed': base_seed,
                        'total_eps': total_eps,
                        'score_type': exp_score
                    }
                    run_new_expo(args)


from email_notifier.email_sender import EmailNotifier

notifier = EmailNotifier()
arguments = {'Esperimento': 'generazione jaccard da 400 a 700'}
notifier.notify(fun, additional_body=str(arguments))
