#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import time
import logging

import numpy as np
cimport numpy as np
from libc.stdint cimport int32_t, int8_t, int64_t


cdef extern from 'learn.c':
    float inplace_update(
        float* In, float* Out,
        int M, int T, double* z,
        int32_t x, int32_t* context, int context_length,
        int32_t* paths, int8_t* codes, int64_t length,
        float* in_grad, float* out_grad,
        float lr, float sense_threshold)

    void update_z(
        float* In, float* Out,
        int M, int T, double* z,
        int32_t x, int32_t* context, int64_t context_length,
        int32_t* paths, int8_t* codes, int64_t length)

    double init_z(double* pi, int T, double alpha, double d, float min_prob)


def inplace_train(
        vm,
        np.ndarray[np.int32_t, ndim=1] doc,
        int32_t window_length,
        float start_lr,
        int64_t total_words,
        int64_t words_read,
        np.ndarray[np.float64_t, ndim=1] total_ll,
        int32_t context_cut,
        float sense_threshold,
        int64_t report_batch_size=10000,
        float min_sense_prob=1e-3):

    cdef np.ndarray[np.float32_t, ndim=3] In = vm.In
    cdef float* In_ptr = &In[0,0,0]

    cdef np.ndarray[np.float32_t, ndim=2] Out = vm.Out
    cdef float* Out_ptr = &Out[0,0]

    cdef np.ndarray[np.int32_t, ndim=2] path = vm.path
    cdef int32_t* path_ptr = &path[0,0]

    cdef np.ndarray[np.int8_t, ndim=2] code = vm.code
    cdef int8_t* code_ptr = &code[0,0]

    cdef np.ndarray[np.float32_t, ndim=2] counts = vm.counts
    cdef np.ndarray[np.int64_t, ndim=1] frequencies = vm.frequencies

    cdef np.ndarray[np.float32_t, ndim=2] in_grad = \
        np.zeros((vm.prototypes, vm.dim), dtype=np.float32)
    cdef float* in_grad_ptr = &in_grad[0,0]

    cdef np.ndarray[np.float32_t, ndim=1] out_grad = np.zeros(vm.dim, dtype=np.float32)
    cdef float* out_grad_ptr = &out_grad[0]

    cdef np.ndarray[np.float64_t, ndim=1] z = np.zeros(vm.prototypes, dtype=np.float64)
    cdef double* z_ptr = &z[0]

    cdef np.ndarray[np.int32_t, ndim=1] context = np.zeros(2 * window_length, dtype=np.int32)
    cdef int32_t* context_ptr = &context[0]

    t0 = time.time()

    cdef double senses = 0.
    cdef double max_senses = 0.
    cdef float min_lr = start_lr * 1e-4
    cdef float lr;
    cdef int32_t window;
    cdef size_t c_len;
    cdef size_t doc_len = len(doc)
    cdef size_t i, k, j;
    cdef double vm_alpha = vm.alpha
    cdef double vm_d = vm.d
    cdef int vm_prototypes = vm.prototypes
    cdef int vm_dim = vm.dim
    cdef size_t path_length = vm.path.shape[1]
    cdef size_t code_length = vm.code.shape[1]
    cdef int32_t w;
    cdef float frequencies_w

    for i in range(doc_len):
        w = doc[i]

        lr = max(start_lr * (1 - words_read / (total_words + 1)), min_lr)
        window = window_length
        if context_cut:
            # TODO - cython optimize
            window -= np.random.randint(1, window_length)

        for k in range(vm_prototypes):
            z[k] = counts[w, k]
        n_senses = init_z(
            z_ptr, vm_prototypes, vm_alpha, vm_d,
            min_sense_prob)
        senses += n_senses
        max_senses = max(max_senses, n_senses)
        c_len = 0
        for j in range(max(0, i - window), min(doc_len, i + window + 1)):
            if i != j:
                context[c_len] = doc[j]
                c_len += 1
        update_z(In_ptr, Out_ptr,
                 vm_dim, vm_prototypes, z_ptr,
                 w, context_ptr, c_len,
                 path_ptr, code_ptr, path_length)

        total_ll[0] += inplace_update(
            In_ptr, Out_ptr,
            vm_dim, vm_prototypes, z_ptr,
            w, context_ptr, c_len,
            path_ptr, code_ptr, code_length,
            in_grad_ptr, out_grad_ptr,
            lr, sense_threshold)
        total_ll[1] += c_len
        words_read += 1

        # variational update for q(pi_v)
        frequencies_w = frequencies[w]
        for k in range(vm_prototypes):
            counts[w, k] += lr * (z[k] * frequencies_w - counts[w, k])

        if i and i % report_batch_size == 0:
            t1 = time.time()
            logging.info(
                '{:.2%} {:.4f} {:.4f} {:.1f}/{:.1f} {:.2f} kwords/sec'
                .format(words_read / total_words, total_ll[0] / total_ll[1],
                        lr, senses / i, max_senses,
                        report_batch_size / 1000 / (t1 - t0)))
            t0 = t1
