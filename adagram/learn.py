from __future__ import absolute_import, division, print_function
import codecs
import logging
import multiprocessing
from multiprocessing.pool import ThreadPool
import time

import numpy as np

from . import clearn


def inplace_train(vm, train_filename, window_length,
                  context_cut=False, epochs=1, sense_threshold=1e-10,
                  batch_size=64000, start_lr=0.025,  encoding='utf8',
                  n_workers=None):
    assert epochs == 1 # TODO - epochs
    total_words = float(vm.frequencies.sum())
    total_ll = np.zeros(2, dtype=np.float64)
    vm.counts[:,0] = vm.frequencies

    def process_item(item):
        words_read, doc = item
        x = clearn.inplace_train(
            vm, doc, window_length, start_lr, total_words, words_read,
            total_ll, context_cut=context_cut, sense_threshold=sense_threshold)
        return (len(doc),) + x

    n_workers = n_workers or multiprocessing.cpu_count()
    with ThreadPool(processes=n_workers) as pool:
        words_processed = wp_since_report = 0
        t0 = time.time()
        for i, (doc_len, lr, senses, max_senses) in enumerate(
                pool.imap_unordered(process_item, _words_reader(
                    vm.dictionary, train_filename, batch_size, encoding)), 1):
            words_processed += doc_len
            wp_since_report += doc_len
            if i % n_workers == 0:
                t1 = time.time()
                logging.info(
                    '{:.2%} {:.4f} {:.4f} {:.1f}/{:.1f} {:.2f} kwords/sec'
                        .format(float(words_processed) / total_words,
                                total_ll[0] / total_ll[1],
                                lr, senses / doc_len, max_senses,
                                wp_since_report / (t1 - t0) / 1000))
                wp_since_report = 0
                t0 = t1


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

