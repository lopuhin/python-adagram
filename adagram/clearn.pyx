#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport log
from libc.stdint cimport int32_t, int8_t, int64_t


cdef extern from 'learn.c':
    float inplace_update(
        float* In, float* Out,
        int M, int T, double* z,
        int32_t x, int32_t* context, int context_length,
        int32_t* paths, int8_t* codes, int64_t length,
        float* in_grad, float* out_grad,
        float lr, float sense_threshold) nogil

    void update_z(
        float* In, float* Out,
        int M, int T, double* z,
        int32_t x, int32_t* context, int64_t context_length,
        int32_t* paths, int8_t* codes, int64_t length) nogil

    float logsigmoid(float x) nogil


def inplace_update_z(vm, np.float64_t[:] z, int32_t w, np.int32_t[:] context):
    cdef np.float32_t[:, :, :] In = vm.In
    cdef np.float32_t[:, :] Out = vm.Out
    cdef np.int32_t[:, :] path = vm.path
    cdef np.int8_t[:, :] code = vm.code
    update_z(&In[0,0,0], &Out[0,0],
             vm.dim, vm.prototypes, &z[0],
             w, &context[0], len(context),
             &path[0,0], &code[0,0], vm.path.shape[1])


def inplace_update_collocates(
        np.int32_t[:, :] path,
        np.int8_t[:, :] code,
        np.float32_t[:] out_dp,
        np.float32_t[:] z_values):
    cdef size_t path_len = path.shape[1]
    cdef size_t n_words = z_values.shape[0]
    cdef np.float32_t f, z
    cdef size_t w_id, n
    cdef int8_t c
    cdef np.int32_t[:] _path
    cdef np.int8_t[:] _code
    with nogil:
        for w_id in range(n_words):
            z = 0.
            _path = path[w_id]
            _code = code[w_id]
            for n in range(path_len):
                c = _code[n]
                if c == -1:
                    break
                f = out_dp[_path[n]]
                z += logsigmoid(f * (1. - 2. * c))
            z_values[w_id] = z


def inplace_train(
        vm,
        np.int32_t[:] doc,
        int32_t window_length,
        float start_lr,
        int64_t total_words,
        int64_t words_read,
        np.float64_t[:] total_ll,
        int32_t context_cut,
        float sense_threshold,
        float min_sense_prob=1e-3):

    cdef np.float32_t[:, :, :] In = vm.In
    cdef np.float32_t[:, :] Out = vm.Out
    cdef np.int32_t[:, :] path = vm.path
    cdef np.int8_t[:, :] code = vm.code

    cdef np.float32_t[:, :] counts = vm.counts
    cdef np.int64_t[:] frequencies = vm.frequencies
    cdef np.float32_t[:, :] in_grad = np.zeros((vm.prototypes, vm.dim), dtype=np.float32)
    cdef np.float32_t[:] out_grad = np.zeros(vm.dim, dtype=np.float32)
    cdef np.float64_t[:] pi = np.zeros(vm.prototypes, dtype=np.float64)
    cdef np.int32_t[:] context = np.zeros(2 * window_length, dtype=np.int32)

    cdef double senses = 0.
    cdef double n_senses
    cdef double max_senses = 0.
    cdef float min_lr = start_lr * 1e-4
    cdef float lr
    cdef int32_t window
    cdef size_t c_len
    cdef size_t doc_len = doc.shape[0]
    cdef size_t i, k, j
    cdef double vm_alpha = vm.alpha
    cdef double vm_d = vm.d
    cdef int vm_prototypes = vm.prototypes
    cdef int vm_dim = vm.dim
    cdef size_t path_length = vm.path.shape[1]
    cdef size_t code_length = vm.code.shape[1]
    cdef int32_t w
    cdef float frequencies_w

    with nogil:
        for i in range(doc_len):
            w = doc[i]

            lr = max(start_lr * (1 - float(words_read) / (total_words + 1)), min_lr)
            window = window_length
            if context_cut:
                with gil:
                    # TODO - cython optimize
                    window -= np.random.randint(1, window_length)

            for k in range(vm_prototypes):
                pi[k] = counts[w, k]
            n_senses = init_z(pi, vm_alpha, vm_d, min_sense_prob)
            senses += n_senses
            max_senses = max(max_senses, n_senses)
            c_len = 0
            for j in range(max(0, i - window), min(doc_len, i + window + 1)):
                if i != j:
                    context[c_len] = doc[j]
                    c_len += 1
            update_z(&In[0,0,0], &Out[0,0],
                     vm_dim, vm_prototypes, &pi[0],
                     w, &context[0], c_len,
                     &path[0,0], &code[0,0], path_length)

            total_ll[0] += inplace_update(
                &In[0,0,0], &Out[0,0],
                vm_dim, vm_prototypes, &pi[0],
                w, &context[0], c_len,
                &path[0,0], &code[0,0], code_length,
                &in_grad[0,0], &out_grad[0],
                lr, sense_threshold)
            total_ll[1] += c_len
            words_read += 1

            # variational update for q(pi_v)
            frequencies_w = frequencies[w]
            for k in range(vm_prototypes):
                counts[w, k] += lr * (pi[k] * frequencies_w - counts[w, k])

    return lr, senses, max_senses


cdef double init_z(
        np.float64_t[:] pi,
        double vm_alpha,
        double vm_d,
        float min_sense_prob) nogil:

    cdef double r = 0.
    cdef double x = 1.
    cdef double a, b, pi_k
    cdef double senses = 0.
    cdef int k
    cdef size_t vm_prototypes = pi.shape[0]

    cdef double ts = 0.
    for k in range(vm_prototypes):
        ts += pi[k]

    for k in range(vm_prototypes - 1):
        ts = max(ts - pi[k], 0.)
        a = 1. + pi[k] - vm_d
        b = vm_alpha + k * vm_d + ts
        pi[k] = meanlog_beta(a, b) + r
        r += meanlog_beta(b, a)

        pi_k = mean_beta(a, b) * x
        x = max(x - pi_k, 0.)
        if pi_k >= min_sense_prob:
            senses += 1.

    pi[vm_prototypes - 1] = r
    if x >= min_sense_prob:
        senses += 1.

    return senses


cdef double mean_beta(double a, double b) nogil:
    return a / (a + b)


cdef double meanlog_beta(double a, double b) nogil:
    return digamma(a) - digamma(a + b)


cdef double digamma(double x) nogil:
    cdef double result = 0., xx, xx2, xx4;
    while x < 7.:
        result -= 1. / x
        x += 1.
    x -= 1. / 2.
    xx = 1. / x
    xx2 = xx * xx
    xx4 = xx2 * xx2
    result += (log(x) + (1. / 24.) * xx2 - (7. / 960.) * xx4 +
               (31. / 8064.) * xx4 * xx2 - (127. / 30720.) * xx4 * xx4)
    return result
