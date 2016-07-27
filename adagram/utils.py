from __future__ import print_function, division
from functools import wraps

import numpy as np


def rand_arr(shape, norm, dtype):
    return (np.array(np.random.rand(*shape), dtype=dtype) - 0.5) * norm


def statprofile(fn):
    import statprof
    @wraps(fn)
    def inner(*args, **kwargs):
        statprof.reset(frequency=1000)
        statprof.start()
        try:
            return fn(*args, **kwargs)
        finally:
            statprof.stop()
            statprof.display()
    return inner
