from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class SpikeTrain(object):

    def __init__(self, spikes, duration):
        """
        Initialize a spike train
        :param spikes: array of times of occurrences of spikes
        :param duration: duration of this train, in seconds
        """
        self.spikes = spikes
        self.duration = duration

    def spike_counts(self, interval):
        """
        Produces the spike counts array of this spike train, given the interval
        :param interval: counting interval, in milliseconds
        :return: an array of spike counts
        """
        interval_sec = interval / 1000.
        return np.diff([np.count_nonzero(self.spikes < t) for t in np.arange(0, self.duration, interval_sec)])

    def interspike_intervals(self):
        """
        Returns the interspike intervals of this spike train
        :return: numpy array
        """
        return np.diff(self.spikes)

    def coefficient_variation(self):
        """
        The Coefficient of variation C_v = sigma_tau / mean_tau
        :return:
        """
        interspike = self.interspike_intervals()
        return np.std(interspike) / np.mean(interspike)

    def fano_factor(self, counting_intervals):
        """
        Compute the Fano factor sigma^2_n / mean_n with every given counting interval
        :param counting_intervals:
        :return:
        """
        ls = []
        for i in counting_intervals:
            counts = self.spike_counts(i)
            ls.append(np.var(counts) / np.mean(counts))
        return np.asarray(ls)

    def interspike_interval_histogram(self, bins):
        """
        Compute the interspike interval histogram: number of intervals falling in discrete time bins.
        :param bins: margins of the bins, in milliseconds
        :return: numpy array
        """
        intervals = self.interspike_intervals()
        return np.diff([np.count_nonzero(intervals < (t / 1000.)) for t in [0] + list(bins)]) / float(intervals.size)

    def autocorrelation_histogram(self, bins):
        """

        :param bins: margins of the bins, in milliseconds
        :return:
        """
        pass