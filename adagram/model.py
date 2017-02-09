from __future__ import absolute_import, division, print_function
import codecs
from collections import Counter
import six
from six.moves import xrange as range

import joblib
import numpy as np
from numpy.linalg import norm

from .learn import inplace_train
from .clearn import inplace_update_z, inplace_update_collocates
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

    def sense_neighbors(self, word, sense, max_neighbors=10,
                        min_closeness=None, min_count=1):
        """ Nearest neighbors of given sense of the word.
        :arg word: word (a string)
        :arg sense: integer sense id (starting from 0)
        :arg max_neighbors: max number of neighbors returned
        :arg min_count: min count of returned neighbors
        :arg min_closeness: min closeness of returned neighbors
        :return: A list of triples (word, sense, closeness)
        """
        if not self.is_valid_sense_vector(word, sense):
            return []
        word_id = self.dictionary.word2id[word]
        s_v = self.In[word_id, sense] / self.InNorms[word_id, sense]
        sim_matrix = np.dot(self.In, s_v) / self.InNorms
        sim_matrix[np.isnan(sim_matrix)] = -np.inf
        most_similar = []
        while True:
            idx = sim_matrix.argmax()
            w_id, s = idx // self.prototypes, idx % self.prototypes
            sim = sim_matrix[w_id, s]
            sim_matrix[w_id, s] = -np.inf
            if ((w_id, s) != (word_id, sense) and
                    self.counts[w_id, s] >= min_count):
                if min_closeness is not None and sim < min_closeness:
                    break
                most_similar.append((self.dictionary.id2word[w_id], s, sim))
            if max_neighbors is not None and len(most_similar) >= max_neighbors:
                break
        return most_similar

    def is_valid_sense_vector(self, word, sense):
        word_id = self.dictionary.word2id[word]
        s_v, s_v_norm = self.In[word_id, sense], self.InNorms[word_id, sense]
        return not (np.allclose(s_v, 0) or np.allclose(s_v_norm, 0))

    def word_sense_collocates(self, word, limit=10, min_prob=1e-3):
        all_z_values = [
            (sense, self.inverse_disambiguate(word, sense))
            for sense, prob in self.word_sense_probs(word)
            if prob >= min_prob and self.is_valid_sense_vector(word, sense)]
        if len(all_z_values) < 2:
            # It's possible to invent something for len = 1
            return []
        z_values_sum = np.zeros_like(all_z_values[0][1])
        for _, z_values in all_z_values:
            z_values_sum += z_values
        z_values_sum /= len(all_z_values)
        return [
            (sense,
             [self.dictionary.id2word[w_id]
              for w_id in (z_values / z_values_sum).argsort()[:limit]])
            for sense, z_values in all_z_values]

    def disambiguate(self, word, context, min_prob=1e-3, use_prior=True):
        """ Return an array of probabilities for each sense of word in context.
        """
        word_idx = self.dictionary.word2id[word]
        if use_prior:
            z = expected_pi(self, word_idx)
            z[z < min_prob] = 0
            z = np.log(z)
        else:
            z = np.zeros(self.prototypes, dtype=np.float64)
        inplace_update_z(
            self, z, word_idx,
            context=np.array(
                [self.dictionary.word2id[w] for w in context
                 if w in self.dictionary.word2id],
                dtype=np.int32))
        return z

    def inverse_disambiguate(self, word, sense):
        """ Run "inverse" disambiguation over the whole vocabulary.
        """
        in_vec = self.sense_vector(word, sense)
        z_values = np.zeros(self.n_words, dtype=np.float32)
        out_dp = np.dot(self.Out, in_vec)
        inplace_update_collocates(self.path, self.code, out_dp, z_values)
        return z_values

    def word_sense_probs(self, word, min_prob=1e-3):
        """ A list of all indices and sense probabilities for given word.
        """
        return [
            (idx, p) for idx, p in enumerate(
                expected_pi(self, self.dictionary.word2id[word]))
            if p >= min_prob]

    def sense_vector(self, word, sense, normalized=False):
        word_id = self.dictionary.word2id[word]
        v = self.In[word_id, sense]
        if normalized:
            nv = self.InNorms[word_id, sense]
            if not np.isclose(nv, 0):
                v = v / nv
        return v

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
        self.In = self.In[:n]
        # Out, path and code can't be slimmed down so easily, leave them as-is.
        self.counts = self.counts[:n]
