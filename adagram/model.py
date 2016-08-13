from __future__ import absolute_import, division, print_function
import codecs
import pickle
from six.moves import xrange as range

import numpy as np
from numpy.linalg import norm

from .softmax import build_huffman_tree, convert_huffman_tree
from .utils import rand_arr


class Dictionary(object):
    def __init__(self, frequencies, id2word):
        assert len(frequencies) == len(id2word)
        self.frequencies = np.array(frequencies, dtype=np.int64)
        self.id2word = id2word
        self.word2id = {w: id_ for id_, w in enumerate(self.id2word)}

    @classmethod
    def read(cls, filename, min_freq, encoding='utf8'):
        words_freqs = []
        with codecs.open(filename, 'rt', encoding=encoding) as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                try:
                    word, freq = line.split()
                    freq = int(freq)
                except ValueError:
                    raise ValueError(
                        u'Expected "word freq" pair on line {}, got "{}"'
                        .format(n, line))
                if freq >= min_freq:
                    words_freqs.append((word, freq))
        words_freqs.sort(key=lambda x: x[1], reverse=True)
        return cls(
            frequencies=[f for _, f in words_freqs],
            id2word=[w for w, _ in words_freqs])

    def __len__(self):
        return len(self.id2word)


class VectorModel(object):
    def __init__(self, frequencies, dim, prototypes, alpha):
        self.alpha = alpha
        self.d = 0.
        self.frequencies = frequencies
        self.prototypes = prototypes
        self.dim = dim
        self.n_words = N = len(self.frequencies)

        nodes = build_huffman_tree(frequencies)
        outputs = convert_huffman_tree(nodes, N)

        max_length = max(len(x.code) for x in outputs)
        self.path = np.zeros((N, max_length), dtype=np.int32)
        self.code = np.zeros((N, max_length), dtype=np.int8)

        for n, output in enumerate(outputs):
            self.code[n] = -1
            for i, (c, p) in enumerate(zip(output.code, output.path)):
                self.code[n, i] = c
                self.path[n, i] = p

        self.In = rand_arr((N, prototypes, dim), 1. / dim, np.float32)
        self.Out = rand_arr((N, dim), 1. / dim, np.float32)
        self.counts = np.zeros((N, prototypes), np.float32)
        self.InNorm = None

    def normalize(self):
        self.InNorm = np.zeros(self.In.shape, dtype=self.In.dtype)
        for w_id in range(self.n_words):
            for s in range(self.prototypes):
                v = self.In[w_id, s]
                self.InNorm[w_id, s] = v / norm(v)


def save_model(output, vm, dictionary):
    with open(output, 'wb') as f:
        pickle.dump((vm, dictionary), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(filename):
    with open(filename, 'rb') as f:
        vm, d = pickle.load(f)
        vm.normalize()
        return vm, d
