import sys
import time

import numpy.random
from numpy import random

import warnings
import numpy as np
import scipy.stats as st


# Create models from data
def best_fit_distribution(data, bins=200):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    d_min = sys.float_info.max
    d_max = 0
    for d_data in data:
        d_min = min(d_min, d_data)
        d_max = max(d_max, d_data)

    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    distributions = [
        st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2,
        st.cosine, st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f,
        st.fatiguelife, st.fisk, st.foldcauchy, st.foldnorm, st.genlogistic, st.genpareto,
        st.gennorm, st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz,
        st.gumbel_r, st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant,
        st.invgamma, st.invgauss, st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace,
        st.levy, st.levy_l, st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke,
        st.nakagami, st.ncx2, st.ncf, st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm,
        st.powernorm, st.rdist, st.reciprocal, st.rayleigh, st.rice, st.semicircular, st.t, st.triang, st.truncexpon,
        st.truncnorm, st.tukeylambda, st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min,
        st.weibull_max, st.wrapcauchy
    ]

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
                warnings.filterwarnings('ignore')

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

    return {"distribution_name": best_distribution.name, "distribution_params": best_params}


def generate_number_from(distribution_name, params):
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    if distribution_name == "fix":
        return arg[0]

    dist = getattr(st, distribution_name)
    num_param = len(arg)

    if num_param == 0:
        return dist.rvs(loc=loc, scale=scale, size=1)[0]
    elif num_param == 1:
        return dist.rvs(arg[0], loc=loc, scale=scale, size=1)[0]
    elif num_param == 2:
        return dist.rvs(arg[0], arg[1], loc=loc, scale=scale, size=1)[0]
    elif num_param == 3:
        return dist.rvs(arg[0], arg[1], arg[2], loc=loc, scale=scale, size=1)[0]


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
                selected.append(self.candidates_list[i])
        return selected if len(selected) > 0 else [self.get_outgoing_flow()]


def random_uniform(start, end):
    return numpy.random.uniform(low=start, high=end)