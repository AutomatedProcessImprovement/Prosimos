import sys
import warnings

import numpy
import numpy as np
import scipy.stats as st
from numpy import random


# NOTE: default distribution becoming obsolete due to no support with pix_framework
def create_default_distribution(min_value, max_value):
    return {"distribution_name": "default", "distribution_params": [min_value, max_value]}


# TODO: consider using get_best_fitting_distribution from pix_framework
# Create models from data
def best_fit_distribution(data, bins=50):
    fix_value = check_fix(data)
    if fix_value is not None:
        return {"distribution_name": "fix", "distribution_params": [fix_value]}

    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    d_min = sys.float_info.max
    d_max = 0
    for d_data in data:
        d_min = min(d_min, d_data)
        d_max = max(d_max, d_data)

    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    distributions = [st.norm, st.expon, st.exponnorm, st.gamma, st.triang, st.uniform, st.lognorm]

    # Discrete distributions
    # disc_distributions = [
    #     st.bernoulli, st.betabinom, st.binom, st.boltzmann, st.planck, st.poisson, st.geom, st.nbinom, st.hypergeom,
    #     st.nchypergeom_fisher, st.nchypergeom_wallenius, st.nhypergeom, st.zipf, st.zipfian, st.logser, st.randint,
    #     st.dlaplace, st.yulesimon, st.norm, st.expon, st.exponnorm, st.gamma, st.triang, st.lognorm, st.uniform
    # ]

    # Distributions to check
    # all_distributions = [
    #     st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2,
    #     st.cosine, st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f,
    #     st.fatiguelife, st.fisk, st.foldcauchy, st.foldnorm, st.genlogistic, st.genpareto,
    #     st.gennorm, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz,
    #     st.gumbel_r, st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant,
    #     st.invgamma, st.invgauss, st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace,
    #     st.levy, st.levy_l, st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke,
    #     st.nakagami, st.ncx2, st.ncf, st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm,
    #     st.powernorm, st.rdist, st.reciprocal, st.rayleigh, st.rice, st.semicircular, st.t, st.triang, st.truncexpon,
    #     st.truncnorm, st.tukeylambda, st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min,
    #     st.weibull_max, st.wrapcauchy
    # ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    i = 1
    for distribution in distributions:
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            # start = time.time()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse
            # end = time.time()
            # print("%d- %s: %.3f" % (i, distribution.name, end - start))
            i += 1
        except Exception:
            pass

    best_params += (d_min, d_max)
    return {"distribution_name": best_distribution.name, "distribution_params": best_params}


def check_fix(data_list, delta=5):
    for d1 in data_list:
        count = 0
        for d2 in data_list:
            if abs(d1 - d2) < delta:
                count += 1
        if count / len(data_list) > 0.9:
            return d1
    return None


def generate_number_from(distribution_name, params):
    while True:
        duration = evaluate_distribution_function(distribution_name, params)
        if duration >= 0:
            return duration


def evaluate_distribution_function(distribution_name, params):
    if distribution_name == "fix":
        return params[0]
    elif distribution_name == "default":
        return numpy.random.uniform(params[0], params[1])

    arg = params[:-4]
    loc = params[-4]
    scale = params[-3]
    d_min = params[-2]
    d_max = params[-1]

    dist = getattr(st, distribution_name)
    num_param = len(arg)

    f_dist = 0
    while True:
        if num_param == 0:
            f_dist = dist.rvs(loc=loc, scale=scale, size=1)[0]
        elif num_param == 1:
            f_dist = dist.rvs(arg[0], loc=loc, scale=scale, size=1)[0]
        elif num_param == 2:
            f_dist = dist.rvs(arg[0], arg[1], loc=loc, scale=scale, size=1)[0]
        elif num_param == 3:
            f_dist = dist.rvs(arg[0], arg[1], arg[2], loc=loc, scale=scale, size=1)[0]
        elif num_param == 4:
            f_dist = dist.rvs(arg[0], arg[1], arg[2], arg[3], loc=loc, scale=scale, size=1)[0]
        elif num_param == 5:
            f_dist = dist.rvs(arg[0], arg[1], arg[2], arg[3], arg[4], loc=loc, scale=scale, size=1)[0]
        elif num_param == 6:
            f_dist = dist.rvs(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], loc=loc, scale=scale, size=1)[0]
        elif num_param == 7:
            f_dist = dist.rvs(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], loc=loc, scale=scale, size=1)[0]
        if d_min <= f_dist <= d_max:
            break
    return f_dist


class Choice:
    def __init__(self, candidates_list, probability_list):
        self.candidates_list = candidates_list
        self.probability_list = probability_list

    def get_outgoing_flow(self):
        return random.choice(self.candidates_list, 1, p=self.probability_list)[0]

    def get_multiple_flows(self):
        selected = list()
        for i in range(0, len(self.candidates_list)):
            if random.choice([True, False], 1, p=[self.probability_list[i], 1 - self.probability_list[i]]):
                selected.append((self.candidates_list[i], None))
        return selected if len(selected) > 0 else [(self.get_outgoing_flow(), None)]


def random_uniform(start, end):
    return random.uniform(low=start, high=end)


# TODO: consider for removal, not being used anywhere
# def best_fit_distribution_1(data):
#     fix_value = check_fix(data)
#     if fix_value is not None:
#         return {"distribution_name": "fix", "distribution_params": [check_fix(data)]}

#     mean = statistics.mean(data)
#     variance = statistics.variance(data)
#     st_dev = statistics.pstdev(data)
#     d_min = min(data)
#     d_max = max(data)

#     dist_candidates = [
#         {"distribution_name": "expon", "distribution_params": [0, mean, d_min, d_max]},
#         {"distribution_name": "norm", "distribution_params": [mean, st_dev, d_min, d_max]},
#         {"distribution_name": "uniform", "distribution_params": [d_min, d_max - d_min, d_min, d_max]},
#         {"distribution_name": "default", "distribution_params": [d_min, d_max]}
#     ]

#     if mean != 0:
#         mean_2 = mean ** 2
#         phi = math.sqrt(variance + mean_2)
#         mu = math.log(mean_2 / phi)
#         sigma = math.sqrt(math.log(phi ** 2 / mean_2))

#         dist_candidates.append({"distribution_name": "lognorm",
#                                 "distribution_params": [sigma, 0, math.exp(mu), d_min, d_max]}, )

#     if mean != 0 and variance != 0:
#         dist_candidates.append({"distribution_name": "gamma",
#                                 "distribution_params": [pow(mean, 2) / variance, 0, variance / mean, d_min, d_max]}, )

#     best_dist = None
#     best_emd = sys.float_info.max
#     for dist_c in dist_candidates:
#         ev_list = list()
#         for i in range(0, len(data)):
#             ev_list.append(evaluate_distribution_function(dist_c["distribution_name"], dist_c["distribution_params"]))

#         emd = wasserstein_distance(data, ev_list)
#         if emd < best_emd:
#             best_emd = emd
#             best_dist = dist_c

#     return best_dist
