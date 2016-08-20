from __future__ import absolute_import, division, print_function
import codecs
import logging
import queue
import threading
import multiprocessing

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

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    doc_queue = queue.Queue(maxsize=n_workers * 2)

    # FIXME - this looks crazy convoluted for such a simple thing?
    def worker(stop_queue):
        should_block = True
        while True:
            try:
                stop_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                should_block = False
            try:
                words_read, doc = doc_queue.get(block=should_block, timeout=1)
            except queue.Empty:
                if should_block:
                    continue
                else:
                    break
            clearn.inplace_train(
                vm, doc, window_length, start_lr, total_words, words_read,
                total_ll,
                context_cut=context_cut, sense_threshold=sense_threshold)

    stop_queues = [queue.Queue() for _ in range(n_workers)]
    worker_threads = [threading.Thread(target=worker, args=[q])
                      for q in stop_queues]
    for thread in worker_threads:
        thread.start()
    for item in _words_reader(
            vm.dictionary, train_filename, batch_size, encoding):
        doc_queue.put(item)
    for q in stop_queues:
        q.put(None)
    for thread in worker_threads:
        thread.join()


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

