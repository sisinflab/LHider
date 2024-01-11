import numpy as np

def none_private(number_selections, epsilon, delta):
    return 1

def naive_comp(number_selections, epsilon, delta):
    eps_per_sample = epsilon / number_selections
    return eps_per_sample

def tight_adv_comp(number_selections, epsilon, delta):
    eps_per_sample = bisection_search_epsilon(number_selections, epsilon, delta, tight_comp)
    return eps_per_sample

def bounded_range_comp(number_selections, epsilon, delta):
    eps_per_sample = bisection_search_epsilon(number_selections, epsilon, delta, dong_comp)
    return eps_per_sample

#Formula for the Kairouz et al bound
def tight_comp(num_steps, eps, delta):
    candidate_epsilons = [0.0, 0.0, 0.0]
    candidate_epsilons[0] = num_steps * eps
    first_term = ((np.expm1(eps)) * num_steps * eps) / (np.exp(eps) + 1)
    inner_term = eps + (np.sqrt(num_steps * (eps ** 2)) / delta)
    candidate_epsilons[1] = first_term + (eps * np.sqrt(2 * num_steps * np.log(inner_term)))
    candidate_epsilons[2] = first_term + (eps * np.sqrt(2 * num_steps * np.log(1 / delta)))
    return np.min(candidate_epsilons)

#Formula for the Dong et al. bound
def dong_comp(num_steps, eps, delta):
    candidate_epsilons = [0.0, 0.0]
    candidate_epsilons[0] = num_steps * eps
    inner_term = eps / (-np.expm1(-eps))
    first_term = num_steps * (inner_term - 1 - np.log(inner_term))
    candidate_epsilons[1] = first_term + np.sqrt(num_steps * 0.5 * (eps ** 2) * np.log(1 / delta))
    return np.min(candidate_epsilons)



#-------------------Other composition formulas I tested
def basic_RDPCDP(number_selections, epsilon, delta):
    eps_per_sample = bisection_search_epsilon(number_selections, epsilon, delta, stenkie_RDP_comp)
    return eps_per_sample

def dwork_adv_comp(number_selections, epsilon, delta):
    eps_per_sample = bisection_search_epsilon(number_selections, epsilon, delta, adv_comp_dwork)
    return eps_per_sample

def adv_comp_dwork(num_steps, eps, delta):
    first_term = 2 * num_steps * np.log(1 / delta)
    second_term = num_steps * eps * (np.expm1(eps))
    return (eps * np.sqrt(first_term)) + second_term

def stenkie_RDP_comp(num_steps, eps, delta):
    alphas = [1 + x / 100.0 for x in range(1, 1000)] + list(range(12, 1200))
    moments = [rdp_to_dp(stenkie(num_steps, eps, alpha), alpha, delta) for alpha in alphas]
    return np.min(moments)

def stenkie(num_steps, epsilon, alpha):
    inner_term_1 = np.sinh(np.float128(alpha * epsilon)) - np.sinh(np.float128((alpha - 1) * epsilon))
    inner_term_2 = np.sinh(epsilon)
    rdp_eps = (num_steps / (alpha - 1)) * (np.log(inner_term_1) - np.log(inner_term_2))
    return rdp_eps

def rdp_to_dp(eps, alpha, delta):
    return eps + (np.log(1 / delta) / (alpha - 1))

#Does a binary search to find the epsilon per run for a given epsilon prime and delta
def bisection_search_epsilon(num_steps, epsilon_prime, delta, com_function, error=0.001):
    eps_min = 0.001
    eps_max = 0.25
    epsilon_prime_min = com_function(num_steps, eps_min, delta)
    while epsilon_prime_min > epsilon_prime: #make sure the min is low enough
        eps_min = eps_min / 2.0
        epsilon_prime_min = com_function(num_steps, eps_min, delta)

    epsilon_prime_max = com_function(num_steps, eps_max, delta)
    while epsilon_prime_max < epsilon_prime: # make sure the max is high enough
        eps_max *= 2.0
        epsilon_prime_max = com_function(num_steps, eps_max, delta)

    #Do binary search to find eps
    approx_eps = np.mean([eps_min, eps_max])
    approx_epsilon_prime = com_function(num_steps, approx_eps, delta)

    while np.abs(approx_epsilon_prime - epsilon_prime) > error:
        if approx_epsilon_prime > epsilon_prime:
            eps_max = approx_eps
        else:
            eps_min = approx_eps
        approx_eps = np.mean([eps_min, eps_max])
        approx_epsilon_prime = com_function(num_steps, approx_eps, delta)
    return approx_eps
