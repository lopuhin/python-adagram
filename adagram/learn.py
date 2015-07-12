from __future__ import print_function, division
import cffi

import numpy as np


ffi = cffi.FFI()
ffi.cdef("""
float inplace_update(float* In, float* Out,
    int M, int T, double* z,
    int x,
    int32_t* path, int8_t* code, int64_t length,
    float* in_grad, float* out_grad,
    float lr, float sense_threshold);
void update_z(float* In, float* Out,
    int M, int T, double* z,
    int x,
    int32_t* path, int8_t* code, int64_t length);
double digamma(double x);
""")

superlib = ffi.verify("""
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float logsigmoid(float x) {
    return -log(1 + exp(-x));
}

#define in_offset(In, x, k, M, T) (In) + (x)*(M)*(T) + (k)*(M)


float inplace_update(float* In, float* Out,
    int M, int T, double* z,
    int x,
    int32_t* path, int8_t* code, int64_t length,
    float* in_grad, float* out_grad,
    float lr, float sense_threshold) {

    float pr = 0;

    for (int k = 0; k < T; ++k)
        for (int i = 0; i < M; ++i)
            in_grad[k*M + i] = 0;

    for (int n = 0; n < length && code[n] != -1; ++n) {
        float* out = Out + path[n]*M;

        for (int i = 0; i < M; ++i)
            out_grad[i] = 0;

        for (int k = 0; k < T; ++k) {
            if (z[k] < sense_threshold) continue;

            float* in = in_offset(In, x, k, M, T);

            float f = 0;
            for (int i = 0; i < M; ++i)
                f += in[i] * out[i];

            pr += z[k] * logsigmoid(f * (1 - 2*code[n]));

            float d = 1 - code[n] - sigmoid(f);
            float g = z[k] * lr * d;

            for (int i = 0; i < M; ++i) {
                in_grad[k*M + i] += g * out[i];
                out_grad[i]      += g * in[i];
            }
        }

        for (int i = 0; i < M; ++i)
            out[i] += out_grad[i];
    }

    for (int k = 0; k < T; ++k) {
        if (z[k] < sense_threshold) continue;
        float* in = in_offset(In, x, k, M, T);
        for (int i = 0; i < M; ++i)
            in[i] += in_grad[k*M + i];
    }

    return pr;
}


void update_z(float* In, float* Out,
    int M, int T, double* z,
    int x,
    int32_t* path, int8_t* code, int64_t length) {

    for (int n = 0; n < length && code[n] != -1; ++n) {
        float* out = Out + path[n]*M;

        for (int k = 0; k < T; ++k) {
            float* in = in_offset(In, x, k, M, T);

            float f = 0;
            for (int i = 0; i < M; ++i)
                f += in[i] * out[i];

            z[k] += logsigmoid(f * (1 - 2*code[n]));
        }
    }
}


double digamma(double x) {
    double result = 0, xx, xx2, xx4;
    assert(x > 0);
    for ( ; x < 7; ++x)
        result -= 1/x;
    x -= 1.0/2.0;
    xx = 1.0/x;
    xx2 = xx*xx;
    xx4 = xx2*xx2;
    result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
    return result;
}

""", libraries=['m'])


TYPES = {
    np.dtype('float32'): 'float',
    np.dtype('float64'): 'double',
    np.dtype('int8'): 'int8_t',
    np.dtype('int32'): 'int32_t',
    }

try:
    import __pypy__
except ImportError:
    np_cast = lambda x: ffi.cast(TYPES[x.dtype] + ' *', x.ctypes.data)
else:
    np_cast = lambda x: ffi.cast(
        TYPES[x.dtype] + ' *', x.data._pypy_raw_address())


def inplace_update(vm, In, Out, w, _w, z, lr, in_grad, out_grad, sense_threshold):
    _w = int(_w)  # https://bitbucket.org/pypy/numpy/issues/36/2d-nparray-does-not-allow-indexing-by
    path = np_cast(vm.path[_w])
    code = np_cast(vm.code[_w])
    return superlib.inplace_update(
        In, Out,
        vm.dim, vm.prototypes, z,
        w,
        path, code, vm.code.shape[1],
        in_grad, out_grad,
        lr, sense_threshold)


def var_update_z(vm, In, Out, w, _w, z):
    _w = int(_w)  # https://bitbucket.org/pypy/numpy/issues/36/2d-nparray-does-not-allow-indexing-by
    path = np_cast(vm.path[_w])
    code = np_cast(vm.code[_w])
    superlib.update_z(
        In, Out,
        vm.dim, vm.prototypes, z, w,
        path, code, vm.path.shape[1])


digamma = superlib.digamma

