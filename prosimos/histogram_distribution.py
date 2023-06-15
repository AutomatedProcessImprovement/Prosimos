import numpy as np


class HistogramDistribution:
  def __init__(self, cdf: list, bin_midpoints: list) -> "HistogramDistribution":
    self.cdf = cdf
    self.bin_midpoints = bin_midpoints

  @staticmethod
  def from_dict(input_dict: dict) -> "HistogramDistribution":
    return HistogramDistribution(
      input_dict["histogram_data"]["cdf"],
      input_dict["histogram_data"]["bin_midpoints"]
    )

  def generate_value(self):
    value = np.random.rand(1)[0]
    value_bin = np.searchsorted(self.cdf, value)
    return self.bin_midpoints[value_bin]
