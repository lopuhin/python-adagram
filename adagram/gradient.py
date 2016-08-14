from __future__ import absolute_import, division, print_function
import codecs
import logging

import numpy as np

from . import clearn


def inplace_train(vm, train_filename, window_length,
        batch_size=64000, start_lr=0.025, context_cut=False, epochs=1,
        sense_threshold=1e-32, encoding='utf8'):
    assert epochs == 1 # TODO - epochs
    total_words = float(vm.frequencies.sum())
    total_ll = np.zeros(2, dtype=np.float64)
    vm.counts[:,0] = vm.frequencies
    for words_read, doc in _words_reader(
            vm.dictionary, train_filename, batch_size, encoding):
        logging.info('{:>8.2%}'.format(words_read / total_words))
        clearn.inplace_train(
            vm, doc, window_length, start_lr, total_words, words_read, total_ll,
            context_cut=context_cut, sense_threshold=sense_threshold)


def _words_reader(dictionary, train_filename, batch_size, encoding):
    idx = 0
    words_read = 0
    doc = np.zeros(batch_size, dtype=np.int32)
    with codecs.open(train_filename, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            for w in line.split():
                try:
                    w_id = dictionary.word2id[w]
                except KeyError:
                    continue
                doc[idx] = w_id
                idx += 1
                if idx == batch_size:
                    yield words_read, doc
                    words_read += idx
                    idx = 0
        yield words_read, doc[:idx]

