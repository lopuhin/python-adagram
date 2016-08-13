from __future__ import absolute_import, division, print_function
from six.moves import xrange as range

import numpy as np


def expected_pi(vm, w, min_prob=1e-3):
    pi = np.zeros(vm.prototypes)
    r = 1.
    ts = vm.counts[w, :].sum()
    for k in range(vm.prototypes - 1):
        ts = max(ts - vm.counts[w, k], 0.)
        a, b = 1. + vm.counts[w, k] - vm.d, vm.alpha + k*vm.d + ts
        pi[k] = mean_beta(a, b) * r
        r = max(r - pi[k], 0.)
    pi[-1] = r
    return pi


def mean_beta(a, b):
    return a / (a + b)
