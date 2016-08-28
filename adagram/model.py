from __future__ import absolute_import, division, print_function
import codecs
from collections import Counter
import six
from six.moves import xrange as range

import joblib
import numpy as np
from numpy.linalg import norm

from .learn import inplace_train
from .softmax import build_huffman_tree, convert_huffman_tree
from .stick_breaking import expected_pi
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
        with codecs.open(filename, 'r', encoding=encoding) as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                try:
                    word, freq = line.split(' ')
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
        with codecs.open(filename, 'r', encoding=encoding) as f:
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

    def train(self, input, window, context_cut=False, epochs=1, n_workers=None,
              sense_threshold=1e-10, normalize_inplace=True):
        """ Train model.
        :arg input: is a path to tokenized text corpus
        :arg window: is window (or half-context) size
        :arg context_cut: randomly reduce context size to speed up training
        :arg epochs: number of iterations across input
        :arg n_workers: number of workers (all cores by default)
        :arg sense_threshold: minimal probability of a meaning to contribute
        into gradients
        :arg normalize_inplace: set to True if you don't want to train the model
        any more after finishing training (it takes less space).
        """
        # TODO - input should be a words iterator
        assert self.In is not self.InNorm, 'training no longer possible'
        inplace_train(
            self, input, window,
            context_cut=context_cut, epochs=epochs, n_workers=n_workers,
            sense_threshold=sense_threshold)
        self.normalize(inplace=normalize_inplace)

    def sense_neighbors(self, word, sense, max_neighbors=10, min_count=1):
        """ Nearest neighbors of given sense of the word.
        :arg word: word (a string)
        :arg sense: integer sense id (starting from 0)
        :arg max_neighbors: max number of neighbors returned
        :arg min_count: min count of returned neighbors
        :return: A list of triples (word, sense, closeness)
        """
        word_id = self.dictionary.word2id[word]
        s_v = self.InNorm[word_id, sense]
        sim_matrix = np.dot(self.InNorm, s_v)
        most_similar = []
        while len(most_similar) < max_neighbors:
            idx = sim_matrix.argmax()
            w_id, s = idx // self.prototypes, idx % self.prototypes
            sim = sim_matrix[w_id, s]
            sim_matrix[w_id, s] = -np.inf
            if ((w_id, s) != (word_id, sense) and
                    self.counts[w_id, s] >= min_count):
                most_similar.append((self.dictionary.id2word[w_id], s, sim))
        return most_similar

    def word_sense_probs(self, word, min_prob=1.e-3):
        """ A list of sense probabilities for given word.
        """
        return [p for p in expected_pi(self, self.dictionary.word2id[word])
                if p >= min_prob]

    def sense_vector(self, word, sense):
        word_id = self.dictionary.word2id[word]
        return self.InNorm[word_id, sense]

    @classmethod
    def load(cls, input):
        return joblib.load(input)

    def save(self, output):
        joblib.dump(self, output)

    def normalize(self, inplace=True):
        self.InNorm = self.In if inplace else \
            np.zeros(self.In.shape, dtype=self.In.dtype)
        for w_id in range(self.n_words):
            for s in range(self.prototypes):
                v = self.In[w_id, s]
                self.InNorm[w_id, s] = v / norm(v)
