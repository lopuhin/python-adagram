from __future__ import print_function, division
import sys
import os.path

import cffi
import numpy as np


ffi = cffi.FFI()
ffi.cdef("""
float inplace_update(float* In, float* Out,
    int M, int T, double* z,
    int x, int y,
    int32_t* path, int8_t* code, int64_t length,
    float* in_grad, float* out_grad,
    float lr, float sense_threshold);
void update_z(float* In, float* Out,
    int M, int T, double* z,
    int x, int y,
    int32_t* path, int8_t* code, int64_t length);
double digamma(double x);
""")

with open(os.path.join(os.path.dirname(__file__), 'learn.c'), 'rb') as f:
    superlib = ffi.verify(
        f.read(),
        libraries=['m'],
        extra_compile_args=['-march=native', '-O3', '-ffast-math'])


TYPES = {
    np.dtype('float32'): 'float',
    np.dtype('float64'): 'double',
    np.dtype('int8'): 'int8_t',
    np.dtype('int32'): 'int32_t',
    }

if '__pypy__' in sys.modules:
    np_cast = lambda x: ffi.cast(
        TYPES[x.dtype] + ' *', x.data._pypy_raw_address())
else:
    np_cast = lambda x: ffi.cast(TYPES[x.dtype] + ' *', x.ctypes.data)


def inplace_update(vm, In, Out, w, _w, path, code, z, lr,
        in_grad, out_grad, sense_threshold):
    return superlib.inplace_update(
        In, Out,
        vm.dim, vm.prototypes, z,
        w, _w,
        path, code, vm.code.shape[1],
        in_grad, out_grad,
        lr, sense_threshold)


def update_z(vm, In, Out, w, _w, path, code, z):
    superlib.update_z(
        In, Out,
        vm.dim, vm.prototypes, z, w, _w,
        path, code, vm.path.shape[1])


digamma = superlib.digamma

