from enum import Enum
from numpy import exp, log, sqrt

class SCIPY_DIST_NAME(Enum):
    EXPONENTIAL = "expon"
    NORMAL = "norm"
    FIXED = "fix"
    UNIFORM = "uniform"
    GAMMA = "gamma"
    TRIANGULAR = "triang"
    LOGNORMAL = "lognorm"


def extract_dist_params(dist_name: SCIPY_DIST_NAME, dist_params):
    if dist_name == SCIPY_DIST_NAME.EXPONENTIAL:
        # input: loc = 0, scale = mean
        return {
            "distribution_name": SCIPY_DIST_NAME.EXPONENTIAL.value,
            "distribution_params": [0, dist_params["arg1"]],
        }
    if dist_name == SCIPY_DIST_NAME.NORMAL:
        # input: loc = mean, scale = standard deviation
        return {
            "distribution_name": SCIPY_DIST_NAME.NORMAL.value,
            "distribution_params": [dist_params["mean"], dist_params["arg1"]],
        }
    if dist_name == SCIPY_DIST_NAME.FIXED:
        return {
            "distribution_name": SCIPY_DIST_NAME.FIXED.value,
            "distribution_params": [dist_params["mean"], 0, 1],
        }
    if dist_name == SCIPY_DIST_NAME.UNIFORM:
        # input: loc = from, scale = to - from
        return {
            "distribution_name": SCIPY_DIST_NAME.UNIFORM.value,
            "distribution_params": [
                dist_params["arg1"],
                dist_params["arg2"] - dist_params["arg1"],
            ],
        }
    if dist_name == SCIPY_DIST_NAME.GAMMA:
        # input: shape, loc=0, scale
        mean, variance = dist_params["mean"], dist_params["arg1"]
        return {
            "distribution_name": SCIPY_DIST_NAME.GAMMA.value,
            "distribution_params": [pow(mean, 2) / variance, 0, variance / mean],
        }
    if dist_name == SCIPY_DIST_NAME.TRIANGULAR:
        # input: c = mode, loc = min, scale = max - min
        return {
            "distribution_name": SCIPY_DIST_NAME.TRIANGULAR.value,
            "distribution_params": [
                dist_params["mean"],
                dist_params["arg1"],
                dist_params["arg2"] - dist_params["arg1"],
            ],
        }
    if dist_name == SCIPY_DIST_NAME.LOGNORMAL:
        mean_2 = dist_params["mean"] ** 2
        variance = dist_params["arg1"]
        phi = sqrt([variance + mean_2])[0]
        mu = log(mean_2 / phi)
        sigma = sqrt([log(phi**2 / mean_2)])[0]

        # input: s = sigma = standard deviation, loc = 0, scale = exp(mu)
        return {
            "distribution_name": SCIPY_DIST_NAME.LOGNORMAL.value,
            "distribution_params": [sigma, 0, exp(mu)],
        }

    return None
