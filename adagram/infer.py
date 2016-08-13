from __future__ import absolute_import, division, print_function

import numpy as np


def nearest_neighbors(vm, word, sense, max_neighbours, min_count):
    word_id = vm.dictionary.word2id[word]
    s_v = vm.InNorm[word_id, sense]
    sim_matrix = np.dot(vm.InNorm, s_v)
    most_similar = []
    while len(most_similar) < max_neighbours:
        idx = sim_matrix.argmax()
        w_id, s = idx // vm.prototypes, idx % vm.prototypes
        sim = sim_matrix[w_id, s]
        sim_matrix[w_id, s] = -np.inf
        if (w_id, s) != (word_id, sense) and vm.counts[w_id, s] >= min_count:
            most_similar.append((vm.dictionary.id2word[w_id], s, sim))
    return most_similar
