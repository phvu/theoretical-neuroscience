from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from spike_train import SpikeTrain


def poisson_constant_rate(duration, rate):
    """
    Poisson process with constant firing rate
    :param rate: constant firing rate
    :param duration: duration
    :return:
    """
    # we don't try to be smart here. Just do a straightforward loop
    times = [0]
    while times[-1] < duration:
        # t_{i+1} = t_i - ln(rand)/r
        times.append(times[-1] - np.log(np.random.rand()) / rate)
    return np.asarray(times[1:-1])


def poisson_time_dependent_rate(duration, rate_fn, max_rate):
    """
    Rejection sampling for Poisson process with time-dependent firing rate
    :param duration:
    :param rate_fn: r_est(time, previous_time)
    :param max_rate:
    :return:
    """
    # implement rejection sampling: first generate with max_rate
    times = poisson_constant_rate(duration, max_rate)

    # rejection
    times_retained = [times[0]]
    for i in range(1, len(times)):
        if rate_fn(times[i], times[i - 1]) / max_rate >= np.random.rand():
            times_retained.append(times[i])
    return np.asarray(times_retained)


class HomogeneousPoissonGenerator(object):
    def __init__(self, rate):
        """
        Initialize this Possion generator with the given rate
        :param rate: firing rate, in Hz
        """
        self.rate = rate

    def generate(self, duration):
        """
        Generate a SpikeTrain of given length
        :param duration: in seconds
        :return: SpikeTrain
        """
        return SpikeTrain(poisson_constant_rate(duration, self.rate), duration)


class HomogeneousPoissonGeneratorWithRefractory(object):

    def __init__(self, max_rate, recovery_time):
        """
        Create homogeneous Poisson generator with refractory
        :param max_rate: in Hz
        :param recovery_time: in milliseconds
        """
        self.max_rate = max_rate
        self.recovery_time = recovery_time

    def generate(self, duration):
        """
        Generate a SpikeTrain of given length
        :param duration: in seconds
        :return: SpikeTrain
        """
        def r_est(t, previous_t):
            return self.max_rate - (self.max_rate * np.exp(-(t - previous_t) * 1000. / self.recovery_time))
        return SpikeTrain(poisson_time_dependent_rate(duration, r_est, self.max_rate), duration)


class PoissonGeneratorWithVariableRate(object):

    def __init__(self, max_rate, rate_fn):
        self.rate_fn = rate_fn
        self.max_rate = max_rate

    def generate(self, duration):
        """

        :param duration:
        :return:
        """
        return SpikeTrain(poisson_time_dependent_rate(duration, lambda x, y: self.rate_fn(x), self.max_rate), duration)
