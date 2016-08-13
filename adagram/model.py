from __future__ import absolute_import, division, print_function
import codecs
from collections import Counter
import pickle
import six
from six.moves import xrange as range

import numpy as np
from numpy.linalg import norm

from .gradient import inplace_train
from .softmax import build_huffman_tree, convert_huffman_tree
from .utils import rand_arr


class Dictionary(object):
    def __init__(self, word_freqs):
        words_freqs = sorted(word_freqs, key=lambda x: x[1], reverse=True)
        self.frequencies = np.array([f for _, f in words_freqs], dtype=np.int64)
        self.id2word = [w for w, _ in words_freqs]
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
        return cls(words_freqs)

    @classmethod
    def build(cls, filename, min_freq, encoding='utf8'):
        with codecs.open(filename, 'rt', encoding=encoding) as f:
            word_freqs = Counter(w for line in f for w in line.split())
        return cls([(w, f) for w, f in six.iteritems(word_freqs)
                    if f >= min_freq])

    def __len__(self):
        return len(self.id2word)


class VectorModel(object):
    def __init__(self, dictionary, dim, prototypes, alpha):
        self.alpha = alpha
        self.d = 0.
        self.dictionary = dictionary
        self.frequencies = dictionary.frequencies
        self.prototypes = prototypes
        self.dim = dim
        self.n_words = N = len(self.frequencies)

        nodes = build_huffman_tree(self.frequencies)
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

    def train(self, input, window, context_cut=False, epochs=1):
        inplace_train(self, input, window,
                      context_cut=context_cut, epochs=epochs)
        self.normalize()

    @classmethod
    def load(cls, input):
        with open(input, 'rb') as f:
            return pickle.load(f)

    def save(self, output):
        # TODO - use joblib
        with open(output, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def normalize(self):
        self.InNorm = np.zeros(self.In.shape, dtype=self.In.dtype)
        for w_id in range(self.n_words):
            for s in range(self.prototypes):
                v = self.In[w_id, s]
                self.InNorm[w_id, s] = v / norm(v)
