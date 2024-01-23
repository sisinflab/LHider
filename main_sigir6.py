from src.jobs.sigir_itemknn import run_new_expo

# definisci i parametri necessari
randomizer = 'randomized'
exp_score = 'manhattan'

# dataset
dataset_name = 'gift'
dataset_type = 'train'


def fun():
    for base_seed in range(100, 5001, 100):
        seed = base_seed
        for eph_phi in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
            for reps in [1, 10, 100, 1000]:
                    seed += 1

                    # run
                    args = {
                        'dataset': dataset_name,
                        'type': dataset_type,
                        'eps_phi': eph_phi,
                        'randomizer': randomizer,
                        'reps': reps,
                        'eps_exp': [0.001, 0.002, 0.004, 0.008],
                        'seed': seed,
                        'base_seed': base_seed,
                        'score_type': exp_score
                    }
                    run_new_expo(args)
fun()