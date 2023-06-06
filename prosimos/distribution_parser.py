from numpy import exp, log, sqrt
from pix_framework.statistics.distribution import DistributionType


def extract_dist_params(dist_name: DistributionType, dist_params, is_min_max_boundaries = False):
    dist_params_res = None

    if dist_name == DistributionType.EXPONENTIAL:
        # input: loc = 0 / shift, scale = mean
        shift = dist_params["exp_shift"] if "exp_shift" in dist_params else 0
        dist_params_res = {
            "distribution_name": DistributionType.EXPONENTIAL.value,
            "distribution_params": [shift, dist_params["arg1"]],
        }
    elif dist_name == DistributionType.NORMAL:
        # input: loc = mean, scale = standard deviation
        dist_params_res = {
            "distribution_name": DistributionType.NORMAL.value,
            "distribution_params": [dist_params["mean"], dist_params["arg1"]],
        }
    elif dist_name == DistributionType.FIXED:
        dist_params_res = {
            "distribution_name": DistributionType.FIXED.value,
            "distribution_params": [dist_params["mean"], 0, 1],
        }
    elif dist_name == DistributionType.UNIFORM:
        # input: loc = from, scale = to - from
        dist_params_res = {
            "distribution_name": DistributionType.UNIFORM.value,
            "distribution_params": [
                dist_params["arg1"],
                dist_params["arg2"] - dist_params["arg1"],
            ],
        }
    elif dist_name == DistributionType.GAMMA:
        # input: shape, loc=0, scale
        mean, variance = dist_params["mean"], dist_params["arg1"]
        dist_params_res = {
            "distribution_name": DistributionType.GAMMA.value,
            "distribution_params": [pow(mean, 2) / variance, 0, variance / mean],
        }
    elif dist_name == DistributionType.TRIANGULAR:
        # input: c = mode, loc = min, scale = max - min
        dist_params_res = {
            "distribution_name": DistributionType.TRIANGULAR.value,
            "distribution_params": [
                dist_params["mean"],
                dist_params["arg1"],
                dist_params["arg2"] - dist_params["arg1"],
            ],
        }
    elif dist_name == DistributionType.LOG_NORMAL:
        mean_2 = dist_params["mean"] ** 2
        variance = dist_params["arg1"]
        phi = sqrt([variance + mean_2])[0]
        mu = log(mean_2 / phi)
        sigma = sqrt([log(phi**2 / mean_2)])[0]

        # input: s = sigma = standard deviation, loc = 0, scale = exp(mu)
        dist_params_res = {
            "distribution_name": DistributionType.LOG_NORMAL.value,
            "distribution_params": [sigma, 0, exp(mu)],
        }

    if (dist_params_res is not None and
        is_min_max_boundaries and
        dist_name not in [DistributionType.FIXED, DistributionType.TRIANGULAR, DistributionType.UNIFORM]
    ):
        # we add min and max boundaries to all distributions
        # except the ones in the list
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
    scipy_dist_name = DistributionType[dist_name]

    return extract_dist_params(scipy_dist_name, dist_params)

def get_scipy_distr(distribution_name, distribution_params):
    """
    Transform distribution function from user-friendly to `Scipy` compliant.
    We transform notation of `mean` and `std dev` to `loc` and `scale` supported by `Scipy`.

    :param distribution_name - name of the distribution func (scipy compliant)
    :param distribution_params - array of values which should be supplied to a distr func
    :return: object with distr name and parameters compliant to be used with scipy library
    """

    dist_name_enum: DistributionType = DistributionType(distribution_name)
    
    return extract_dist_params(
        dist_name_enum,
        map_dist_params(distribution_params, dist_name_enum),
        True
    )

def map_dist_params(params, dist_name_enum: DistributionType):
    """
    Transform array of values to the array compliant with what BIMP produces
    """
    if dist_name_enum == DistributionType.UNIFORM:
        # no mean, min and max values
        return {
            "arg1": float(params[0]["value"]),
            "arg2": float(params[1]["value"]), 
        }
    elif dist_name_enum == DistributionType.EXPONENTIAL:
        return {
            "arg1": float(params[0]["value"]),
            "exp_shift": float(params[1]["value"]),
            "min": float(params[2]["value"]),
            "max": float(params[3]["value"])
        }
    elif dist_name_enum == DistributionType.FIXED:
        return {
            "mean": float(params[0]["value"])
        }
    elif dist_name_enum == DistributionType.TRIANGULAR:
        return {
            "mean": float(params[0]["value"]),
            "arg1": float(params[1]["value"]),
            "arg2": float(params[2]["value"])
        }
    elif dist_name_enum in [DistributionType.NORMAL, DistributionType.GAMMA, DistributionType.LOG_NORMAL]:
        return {
            "mean": float(params[0]["value"]),
            "arg1": float(params[1]["value"]),
            "min": float(params[2]["value"]),
            "max": float(params[3]["value"])
        }
