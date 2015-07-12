from __future__ import print_function, division
import numpy as np


def expected_pi(vm, w, min_prob=1e-3):
    pi = np.zeros(vm.prototypes)
    r = 1.
    ts = vm.counts[w, :].sum()
    for k in xrange(vm.prototypes - 1):
        ts = max(ts - vm.counts[w, k], 0.)
        a, b = 1. + vm.counts[w, k] - vm.d, vm.alpha + k*vm.d + ts
        pi[k] = mean_beta(a, b) * r
        r = max(r - pi[k], 0.)
    pi[-1] = r
    return pi


mean_beta = lambda a, b: a / (a + b)
