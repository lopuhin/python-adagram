from __future__ import print_function, division
import numpy as np


# TODO - move to C
def expected_logpi(vm, pi, w, min_prob=1e-3):
    r = 0.
    x = 1.
    senses = 0
    pi[:] = np.array(vm.counts[w, :])
    ts = pi.sum()
    for k in xrange(vm.prototypes - 1):
        ts = max(ts - pi[k], 0.)
        a, b = 1. + pi[k] - vm.d, vm.alpha + k*vm.d + ts
        pi[k] = meanlog_beta(a, b) + r
        r += meanlog_beta(b, a)

        pi_k = mean_beta(a, b) * x
        x = max(x - pi_k, 0.)
        if pi_k >= min_prob:
            senses += 1

    pi[-1] = r
    if x >= min_prob:
        senses += 1
    return senses


def expected_pi(vm, w, min_prob=1e-3):
    pi = np.zeros(vm.prototypes)
    r = 1.
   #senses = 0
    ts = vm.counts[w, :].sum()
    for k in xrange(vm.prototypes - 1):
        ts = max(ts - vm.counts[w, k], 0.)
        a, b = 1. + vm.counts[w, k] - vm.d, vm.alpha + k*vm.d + ts
        pi[k] = mean_beta(a, b) * r
   #    if pi[k] >= min_prob:
   #        senses += 1
        r = max(r - pi[k], 0.)
    pi[-1] = r
   #if r >= min_prob:
   #    senses += 1
    return pi


mean_beta = lambda a, b: a / (a + b)
meanlog_beta = lambda a, b: digamma(a) - digamma(a + b)


try:
    from scipy.special import digamma
except ImportError:
    from adagram.learn import digamma
