from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt

from common import spike_generators


def p1():
    rate = 100
    duration = 10
    spikes = spike_generators.HomogeneousPoissonGenerator(rate).generate(duration)
    print("Coefficient of variation: {}".format(spikes.coefficient_variation()))
    print("Fano factor for counting intervals in [0, 100] ms: {}".format(spikes.fano_factor(np.arange(1, 100))))

    x = np.arange(1, 100)
    y = spikes.interspike_interval_histogram(x)
    plt.bar(x, y, width=1, bottom=0)
    plt.show()


if __name__ == '__main__':
    p1()
