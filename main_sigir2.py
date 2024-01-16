from src.jobs.sigir import run, run_explicit
from composition import tight_adv_comp


# definisci i parametri necessari
randomizer = 'discretized'
eps_phi: float = 1.0
reps = 5
eps_exp: float = 1.0
exp_score = 'manhattan'
total_eps = eps_phi * reps + eps_exp

print('eps_phi', eps_phi,
      'reps', reps,
      'eps_exp', eps_exp,
      'total eps', total_eps)

# riproducibilità
base_seed = 42
seed = base_seed

# dataset
dataset_name = 'yahoo_movies'
dataset_type = 'train'

for base_seed in range(100, 101, 100):
    for eph_phi in [1, 2, 3, 5, 10, 30]:
        for reps in [1, 2, 5, 10, 50, 100]:
            for eps_exp in [1, 3, 5, 10, 100]:
                seed += 1

                items = 1034
                delta = (1 / items) ** 1.1

#                new_eps = tight_adv_comp(desired_gens, desired_eps, delta)

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
                    'total_eps': eph_phi
                }
                run_explicit(args)
