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


def extract_dist_params(dist_name: SCIPY_DIST_NAME, dist_params, is_min_max_boundaries = False):
    dist_params_res = None

    if dist_name == SCIPY_DIST_NAME.EXPONENTIAL:
        # input: loc = 0, scale = mean
        dist_params_res = {
            "distribution_name": SCIPY_DIST_NAME.EXPONENTIAL.value,
            "distribution_params": [0, dist_params["arg1"]],
        }
    elif dist_name == SCIPY_DIST_NAME.NORMAL:
        # input: loc = mean, scale = standard deviation
        dist_params_res = {
            "distribution_name": SCIPY_DIST_NAME.NORMAL.value,
            "distribution_params": [dist_params["mean"], dist_params["arg1"]],
        }
    elif dist_name == SCIPY_DIST_NAME.FIXED:
        dist_params_res = {
            "distribution_name": SCIPY_DIST_NAME.FIXED.value,
            "distribution_params": [dist_params["mean"], 0, 1],
        }
    elif dist_name == SCIPY_DIST_NAME.UNIFORM:
        # input: loc = from, scale = to - from
        dist_params_res = {
            "distribution_name": SCIPY_DIST_NAME.UNIFORM.value,
            "distribution_params": [
                dist_params["arg1"],
                dist_params["arg2"] - dist_params["arg1"],
            ],
        }
    elif dist_name == SCIPY_DIST_NAME.GAMMA:
        # input: shape, loc=0, scale
        mean, variance = dist_params["mean"], dist_params["arg1"]
        dist_params_res = {
            "distribution_name": SCIPY_DIST_NAME.GAMMA.value,
            "distribution_params": [pow(mean, 2) / variance, 0, variance / mean],
        }
    elif dist_name == SCIPY_DIST_NAME.TRIANGULAR:
        # input: c = mode, loc = min, scale = max - min
        dist_params_res = {
            "distribution_name": SCIPY_DIST_NAME.TRIANGULAR.value,
            "distribution_params": [
                dist_params["mean"],
                dist_params["arg1"],
                dist_params["arg2"] - dist_params["arg1"],
            ],
        }
    elif dist_name == SCIPY_DIST_NAME.LOGNORMAL:
        mean_2 = dist_params["mean"] ** 2
        variance = dist_params["arg1"]
        phi = sqrt([variance + mean_2])[0]
        mu = log(mean_2 / phi)
        sigma = sqrt([log(phi**2 / mean_2)])[0]

        # input: s = sigma = standard deviation, loc = 0, scale = exp(mu)
        dist_params_res = {
            "distribution_name": SCIPY_DIST_NAME.LOGNORMAL.value,
            "distribution_params": [sigma, 0, exp(mu)],
        }

    if (dist_params_res is not None and
        is_min_max_boundaries and
        dist_name not in [SCIPY_DIST_NAME.FIXED, SCIPY_DIST_NAME.TRIANGULAR, SCIPY_DIST_NAME.UNIFORM]
    ):
        min, max = dist_params["min"], dist_params["max"]
        dist_params_res["distribution_params"].extend([min, max])

    return dist_params_res

def extract_dist_params_from_qbp(dist_info):
    # time_unit = dist_info.find("qbp:timeUnit", simod_ns).text
    # The time_tables produced by bimp always have the parameters in seconds, although it shows other time units in
    # the XML file.
    dist_params = {
        "mean": float(dist_info.attrib["mean"]),
        "arg1": float(dist_info.attrib["arg1"]),
        "arg2": float(dist_info.attrib["arg2"]),
    }
    dist_name = dist_info.attrib["type"].upper()
    
    # transform distribution names to the format supported by Scipy
    # e.g. "EXPONENTIAL" -> "expon" 
    scipy_dist_name = SCIPY_DIST_NAME[dist_name]

    return extract_dist_params(scipy_dist_name, dist_params)

def get_scipy_distr(distribution_name, distribution_params):
    """
    Transform distribution function from user-friendly to `Scipy` compliant.
    We transform notation of `mean` and `std dev` to `loc` and `scale` supported by `Scipy`.

    :param distribution_name - name of the distribution func (scipy compliant)
    :param distribution_params - array of values which should be supplied to a distr func
    :return: object with distr name and parameters compliant to be used with scipy library
    """

    dist_name_enum: SCIPY_DIST_NAME = SCIPY_DIST_NAME(distribution_name)
    
    return extract_dist_params(
        dist_name_enum,
        map_dist_params(distribution_params, dist_name_enum),
        True
    )

def map_dist_params(params, dist_name_enum: SCIPY_DIST_NAME):
    """
    Transform array of values to the array compliant with what BIMP produces
    """
    if dist_name_enum == SCIPY_DIST_NAME.UNIFORM:
        # no mean, min and max values
        return {
            "arg1": float(params[0]["value"]),
            "arg2": float(params[1]["value"]), 
        }
    elif dist_name_enum == SCIPY_DIST_NAME.EXPONENTIAL:
        return {
            "arg1": float(params[0]["value"]),
            "min": float(params[1]["value"]),
            "max": float(params[2]["value"])
        }
    elif dist_name_enum == SCIPY_DIST_NAME.FIXED:
        return {
            "mean": float(params[0]["value"])
        }
    elif dist_name_enum == SCIPY_DIST_NAME.TRIANGULAR:
        return {
            "mean": float(params[0]["value"]),
            "arg1": float(params[1]["value"]),
            "arg2": float(params[2]["value"])
        }
    elif dist_name_enum in [SCIPY_DIST_NAME.NORMAL, SCIPY_DIST_NAME.GAMMA, SCIPY_DIST_NAME.LOGNORMAL]:
        return {
            "mean": float(params[0]["value"]),
            "arg1": float(params[1]["value"]),
            "min": float(params[2]["value"]),
            "max": float(params[3]["value"])
        }
