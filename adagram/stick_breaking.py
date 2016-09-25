from __future__ import absolute_import, division, print_function
from six.moves import xrange as range

import numpy as np


def expected_pi(vm, w_idx):
    pi = np.zeros(vm.prototypes, dtype=np.float64)
    r = 1.
    ts = vm.counts[w_idx, :].sum()
    for k in range(vm.prototypes - 1):
        ts = max(ts - vm.counts[w_idx, k], 0.)
        a = 1. + vm.counts[w_idx, k] - vm.d
        b = vm.alpha + k * vm.d + ts
        pi[k] = mean_beta(a, b) * r
        r = max(r - pi[k], 0.)
    pi[-1] = r
    return pi


def mean_beta(a, b):
    return a / (a + b)
