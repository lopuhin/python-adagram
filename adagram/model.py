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
    def __init__(self, words_freqs, preserve_indices=False):
        if not preserve_indices:
            words_freqs = sorted(
                words_freqs, key=lambda x: (x[1], x[0]), reverse=True)
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

    def slim_down(self, n):
        self.frequencies = self.frequencies[:n]
        self.id2word = self.id2word[:n]
        self.word2id = {w: id_ for id_, w in enumerate(self.id2word)}


class VectorModel(object):
    def __init__(self, dictionary, dim, prototypes, alpha):
        self.alpha = alpha  # type: float
        self.d = 0.
        self.dictionary = dictionary  # type: Dictionary
        self.frequencies = dictionary.frequencies
        self.prototypes = prototypes  # type: int
        self.dim = dim  # type: int
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

    @property
    def InNorms(self):
        if not hasattr(self, '_InNorms'):
            self._InNorms = norm(self.In, axis=2)
        return self._InNorms

    def train(self, input, window, context_cut=False, epochs=1, n_workers=None,
              sense_threshold=1e-10):
        """ Train model.
        :arg input: is a path to tokenized text corpus
        :arg window: is window (or half-context) size
        :arg context_cut: randomly reduce context size to speed up training
        :arg epochs: number of iterations across input
        :arg n_workers: number of workers (all cores by default)
        :arg sense_threshold: minimal probability of a meaning to contribute
        into gradients
        """
        # TODO - input should be a words iterator
        inplace_train(
            self, input, window,
            context_cut=context_cut, epochs=epochs, n_workers=n_workers,
            sense_threshold=sense_threshold)

    def sense_neighbors(self, word, sense, max_neighbors=10, min_count=1):
        """ Nearest neighbors of given sense of the word.
        :arg word: word (a string)
        :arg sense: integer sense id (starting from 0)
        :arg max_neighbors: max number of neighbors returned
        :arg min_count: min count of returned neighbors
        :return: A list of triples (word, sense, closeness)
        """
        word_id = self.dictionary.word2id[word]
        s_v = self.In[word_id, sense] / self.InNorms[word_id, sense]
        sim_matrix = np.dot(self.In, s_v) / self.InNorms
        sim_matrix[np.isnan(sim_matrix)] = -np.inf
        most_similar = []
        # TODO - check if the loop below needs optimizing
        while len(most_similar) < max_neighbors:
            idx = sim_matrix.argmax()
            w_id, s = idx // self.prototypes, idx % self.prototypes
            sim = sim_matrix[w_id, s]
            sim_matrix[w_id, s] = -np.inf
            if ((w_id, s) != (word_id, sense) and
                    self.counts[w_id, s] >= min_count):
                most_similar.append((self.dictionary.id2word[w_id], s, sim))
        return most_similar

    def sense_collocates(self, word, sense):
        # FIXME: WIP, maybe this is not correct.
        # The idea is to do "disambiguation" across the whole vocabulary,
        # and find out which words are likely to indicate given sense.
        # But this gives strange results.
        in_vec = self.sense_vector(word, sense)
        logsigmoid = lambda x: -np.log(1 + np.exp(-x))
        z_values = np.zeros(self.n_words, dtype=np.float32)
        out_dp = np.dot(self.Out, in_vec)
        for w_id in range(self.n_words):
            path = self.path[w_id]
            code = self.code[w_id]
            z = 0.
            for n in range(path.shape[0]):
                if code[n] == -1:
                    break
                f = out_dp[path[n]]
                z += logsigmoid(f * (1. - 2. * code[n]))
            z_values[w_id] = z
        return z_values

    def word_sense_probs(self, word, min_prob=1.e-3):
        """ A list of sense probabilities for given word.
        """
        return [p for p in expected_pi(self, self.dictionary.word2id[word])
                if p >= min_prob]

    def sense_vector(self, word, sense):
        word_id = self.dictionary.word2id[word]
        return self.In[word_id, sense]

    @classmethod
    def load(cls, input):
        return joblib.load(input)

    def save(self, output):
        joblib.dump(self, output)

    def slim_down(self, n):
        """ Make model smaller: leave only first n words.
        """
        self.dictionary.slim_down(n)
        self.n_words = n
        # FIXME - Out, path and code are wrong after slim,
        # could as well set them to None.
        self.path = self.path[:n]
        self.code = self.code[:n]
        self.In = self.In[:n]
        self.Out = self.Out[:n]
        self.counts = self.counts[:n]
