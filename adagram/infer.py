from __future__ import print_function, division

import numpy as np


def nearest_neighbors(vm, dictionary, word, sense, max_neighbours=10):
    w_id = dictionary.word2id[word]
    s_v = vm.InNorm[w_id, sense]
    return _nearest_neighbors(
        vm, dictionary, s_v, max_neighbours,
        exclude=(w_id, sense))


def _nearest_neighbors(vm, dictionary, s_v, max_neighbours,
        exclude, min_count=1):
    sim_matrix = np.dot(vm.InNorm, s_v)
    most_similar = []
    while len(most_similar) < max_neighbours:
        idx = sim_matrix.argmax()
        w_id, s = idx // vm.prototypes, idx % vm.prototypes
        sim = sim_matrix[w_id, s]
        sim_matrix[w_id, s] = -np.inf
        if (w_id, s) != exclude and vm.counts[w_id, s] > min_count:
            most_similar.append((dictionary.id2word[w_id], s, sim))
    return most_similar
