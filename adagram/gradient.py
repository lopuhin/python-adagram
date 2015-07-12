from __future__ import print_function, division
import random
import time

import numpy as np

from adagram.learn import update_z, inplace_update, np_cast, init_z
from adagram.utils import statprofile


def inplace_train(vm, dictionary, train_filename, window_length,
        batch_size=64000, start_lr=0.025, context_cut=True, epochs=1,
        sense_threshold=1e-32):
    # FIXME - epochs
    total_words = float(dictionary.frequencies.sum())
    total_ll = [0.0, 0.0]
    vm.counts[:,0] = vm.frequencies
    for words_read, doc in _words_reader(
            dictionary, train_filename, batch_size):
        print('{:>8.2%}'.format(words_read / total_words))
        _inplace_train(
            vm, doc, window_length, start_lr, total_words, words_read, total_ll,
            context_cut=context_cut, sense_threshold=sense_threshold)


def _inplace_train(vm, doc, window_length, start_lr, total_words, words_read,
        total_ll, context_cut, sense_threshold,
        report_batch_size=10000, min_sense_prob=1e-3):
    in_grad = np.zeros((vm.prototypes, vm.dim), dtype=np.float32)
    out_grad = np.zeros(vm.dim, dtype=np.float32)
    z = np.zeros(vm.prototypes, dtype=np.float64)
    context = np.zeros(2 * window_length, dtype=np.int32)
    senses = 0.
    max_senses = 0.
    min_lr = start_lr * 1e-4
    t0 = time.time()
    In_ptr = np_cast(vm.In)
    Out_ptr = np_cast(vm.Out)
    z_ptr = np_cast(z)
    in_grad_ptr = np_cast(in_grad)
    out_grad_ptr = np_cast(out_grad)
    path_ptr = np_cast(vm.path)
    code_ptr = np_cast(vm.code)
    context_ptr = np_cast(context)
    for i, w in enumerate(doc):
        lr = max(start_lr * (1 - words_read / (total_words + 1)), min_lr)
        window = window_length
        if context_cut:
            window -= random.randint(1, window_length - 1)

        z[:] = vm.counts[w, :]
        n_senses = init_z(
            z_ptr, vm.prototypes, vm.alpha, vm.d,
            min_sense_prob)
        senses += n_senses
        max_senses = max(max_senses, n_senses)
        c_len = 0
        for j in xrange(max(0, i - window), min(len(doc), i + window + 1)):
            if i != j:
                context[c_len] = doc[j]
                c_len += 1
        update_z(In_ptr, Out_ptr,
                 vm.dim, vm.prototypes, z_ptr,
                 w, context_ptr, c_len,
                 path_ptr, code_ptr, vm.path.shape[1])

        total_ll[0] += inplace_update(
            In_ptr, Out_ptr,
            vm.dim, vm.prototypes, z_ptr,
            w, context_ptr, c_len,
            path_ptr, code_ptr, vm.code.shape[1],
            in_grad_ptr, out_grad_ptr,
            lr, sense_threshold)
        total_ll[1] += len(context)
        words_read += 1

        # variational update for q(pi_v)
        _var_update_counts(vm, w, z, lr)

        if i and i % report_batch_size == 0:
            t1 = time.time()
            print('{:.2%} {:.4f} {:.4f} {:.1f}/{:.1f} {:.2f} kwords/sec'\
                .format(words_read / total_words, total_ll[0] / total_ll[1],
                        lr, senses / i, max_senses,
                        report_batch_size / 1000 / (t1 - t0)))
            t0 = t1


def _var_update_counts(vm, w, z, lr):
    counts = vm.counts[w, :]
    freq = vm.frequencies[w]
    for k in xrange(vm.prototypes):
        counts[k] += lr * (z[k] * freq - counts[k])


def _words_reader(dictionary, train_filename, batch_size):
    idx = 0
    words_read = 0
    doc = np.zeros(batch_size, dtype=np.int32)
    with open(train_filename, 'rb') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            for w in line.split():
                try: w_id = dictionary.word2id[w]
                except KeyError: continue
                doc[idx] = w_id
                idx += 1
                if idx == batch_size:
                    yield words_read, doc
                    words_read += idx
                    idx = 0
        yield words_read, doc[:idx]

