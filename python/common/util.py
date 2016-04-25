from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt


def plot_spikes(spikes, figsize=(200, 1), title='Spike Train'):
    """
    Plot all the spikes in the given spike train
    :param spikes: SpikeTrain object
    :param figsize:
    :param title:
    :return:
    """
    plt.figure(figsize=figsize)
    plt.vlines(spikes.spikes, 0, 1)
    plt.title(title, loc='left')
    plt.show()


def plot_interspike_interval_histogram(spikes, min_time=1, max_time=100,
                                       figsize=None, title='Interspike interval histogram'):
    x = np.arange(min_time, max_time)
    y = spikes.interspike_interval_histogram(x)
    plt.figure(figsize=figsize)
    plt.bar(x, y, width=1, bottom=0)
    plt.title(title, loc='left')
