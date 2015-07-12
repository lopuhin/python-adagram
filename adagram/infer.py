import heapq

import numpy as np
from numpy.linalg import norm


def nearest_neighbors(vm, dictionary, word, sense, max_neighbours=10):
    w_id = dictionary.word2id[word]
    s_v = sense_vector(vm, w_id, sense)
    return _nearest_neighbors(
        vm, dictionary, s_v, max_neighbours,
        exclude=(w_id, sense))


def sense_vector(vm, w_id, sense):
    w_v = vm.In[w_id, sense]
    return w_v / norm(w_v)


def _nearest_neighbors(vm, dictionary, s_v, max_neighbours,
        exclude, min_count=1):
    def sim_iter():
        for w_id in xrange(vm.n_words):
            for s in xrange(vm.prototypes):
                if (w_id, s) != exclude and vm.counts[w_id, s] > min_count:
                    in_vs = vm.In[w_id, s]
                    yield (np.dot(in_vs, s_v) / norm(in_vs), w_id, s)
    return [(dictionary.id2word[w_id], s, sim)
            for sim, w_id, s in heapq.nlargest(max_neighbours, sim_iter())]
