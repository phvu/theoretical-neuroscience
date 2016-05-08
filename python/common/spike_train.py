from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class SpikeTrain(object):

    def __init__(self, spikes, duration):
        """
        Initialize a spike train
        :param spikes: array of times of occurrences of spikes, in seconds
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

    def autocorrelation(self, bin_size, bin_count):
        """

        :param bin_size: int, size of a bin in the histogram, in seconds
        :param bin_count: int, number of bins
        :return: x, y
        """
        # page 28 textbook
        vals = [0] * bin_count
        for t1 in self.spikes:
            for t2 in self.spikes:
                m = int(np.floor(np.abs(t1 - t2) / bin_size))
                if m < bin_count:
                    vals[m] += 1
        v = (np.asarray(vals, dtype=np.float) / self.duration)
        v -= (len(self.spikes) * len(self.spikes) * bin_size) / (self.duration * self.duration)
        return np.arange(0, bin_count) * int(bin_size * 1000), v
