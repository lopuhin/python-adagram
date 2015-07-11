import sys

import numpy as np

from softmax import build_huffman_tree, convert_huffman_tree


class Dictionary(object):
    def __init__(self, frequencies, id2word):
        assert len(frequencies) == len(id2word)
        self.frequencies = np.array(frequencies, dtype=np.int64)
        self.id2word = id2word
        self.word2id = {w: id_ for id_, w in enumerate(self.id2word)}

    @classmethod
    def read(cls, filename, min_freq):
        words_freqs = []
        with open(filename, 'rb') as f:
            for n, line in enumerate(f, 1):
                line = line.decode('utf-8').strip()
                try:
                    word, freq = line.split()
                    freq = int(freq)
                except ValueError:
                    print >>sys.stderr, \
                        u'Expected "word freq" pair on line {}, got "{}"'\
                        .format(n, line)
                    sys.exit(1)
                if freq >= min_freq:
                    words_freqs.append((word, freq))
        words_freqs.sort(key=lambda (w, f): f, reverse=True)
        return cls([f for _, f in words_freqs], [w for w, _ in words_freqs])

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
        self.path = np.zeros((max_length, N), dtype=np.int32)
        self.code = np.zeros((max_length, N), dtype=np.int8)

        for n, output in enumerate(outputs):
            self.code[:, n] = -1
            for i, (c, p) in enumerate(zip(output.code, output.path)):
                self.code[i, n] = c
                self.path[i, n] = p

        self.In = rand_arr((dim, prototypes, N), dim)
        self.Out = rand_arr((dim, N), dim)
        self.counts = np.zeros((prototypes, N), np.float32)


def rand_arr(shape, inv_norm):
    return (np.random.rand(*shape) - 0.5) / inv_norm
