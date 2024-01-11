from composition import tight_adv_comp

desired_eps = 1
desired_gens = 100
items = 1034
delta = (1/items)**1.1

print(tight_adv_comp(desired_gens, desired_eps, delta))