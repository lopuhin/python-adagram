import cffi

import numpy as np


ffi = cffi.FFI()
ffi.cdef("""
float inplace_update(float* In, float* Out,
    int M, int T, double* z,
    int x,
    int32_t* path, int8_t* code, int64_t length,
    float* in_grad, float* out_grad,
    float lr, float sense_treshold);
void update_z(float* In, float* Out,
    int M, int T, double* z,
    int x,
    int32_t* path, int8_t* code, int64_t length);
""")

superlib = ffi.verify("""
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float logsigmoid(float x) {
    return -log(1 + exp(-x));
}

#define in_offset(In, x, k, M, T) (In) + (x)*(M)*(T) + (k)*(M)


//assuming everything is indexed from 1 like in julia
float inplace_update(float* In, float* Out,
    int M, int T, double* z,
    int x,
    int32_t* path, int8_t* code, int64_t length,
    float* in_grad, float* out_grad,
    float lr, float sense_treshold) {

    float pr = 0;

    for (int k = 0; k < T; ++k)
        for (int i = 0; i < M; ++i)
            in_grad[k*M + i] = 0;

    for (int n = 0; n < length && code[n] != -1; ++n) {
        float* out = Out + path[n]*M;

        for (int i = 0; i < M; ++i)
            out_grad[i] = 0;

        for (int k = 0; k < T; ++k) {
            if (z[k] < sense_treshold) continue;

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
        if (z[k] < sense_treshold) continue;
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
    _np_cast = lambda x: ffi.cast(TYPES[x.dtype] + ' *', x.ctypes.data)
else:
    _np_cast = lambda x: ffi.cast(
        TYPES[x.dtype] + ' *', x.data._pypy_raw_address())


def inplace_update(vm, w, _w, z, lr, in_grad, out_grad, sense_treshold):
    _w = int(w)  # https://bitbucket.org/pypy/numpy/issues/36/2d-nparray-does-not-allow-indexing-by
    return superlib.inplace_update(
        _np_cast(vm.In), _np_cast(vm.Out),
        vm.dim, vm.prototypes, _np_cast(z),
        w,
        _np_cast(vm.path[_w]), _np_cast(vm.code[_w]), vm.code.shape[1],
        _np_cast(in_grad), _np_cast(out_grad),
        lr, sense_treshold)


def var_update_z(vm, w, _w, z):
    _w = int(w)  # https://bitbucket.org/pypy/numpy/issues/36/2d-nparray-does-not-allow-indexing-by
    superlib.update_z(
        _np_cast(vm.In), _np_cast(vm.Out),
        vm.dim, vm.prototypes, _np_cast(z), w,
        _np_cast(vm.path[_w]), _np_cast(vm.code[_w]), vm.path.shape[1])



