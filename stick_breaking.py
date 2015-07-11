import numpy as np


def expected_logpi(vm, pi, w, min_prob=1e-3):
    r = 0.
    x = 1.
    senses = 0
    pi[:] = np.array(vm.counts[:, w])
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


mean_beta = lambda a, b: a / (a + b)
meanlog_beta = lambda a, b: digamma(a) - digamma(a + b)


def digamma(x):
    # see http://web.science.mq.edu.au/~mjohnson/code/digamma.c
    result = 0.
    assert x > 0
    while x < 7:
        result -= 1/x
        x += 1
    x -= 1.0/2.0
    xx = 1.0/x
    xx2 = xx*xx
    xx4 = xx2*xx2
    result += np.log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-\
              (127.0/30720.0)*xx4*xx4
    return result



