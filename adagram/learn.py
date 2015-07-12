from __future__ import print_function, division
import sys
import os.path

import cffi
import numpy as np


ffi = cffi.FFI()
ffi.cdef("""
float inplace_update(float* In, float* Out,
    int M, int T, double* z,
    int32_t x, int32_t* context, int context_length,
    int32_t* paths, int8_t* codes, int64_t length,
    float* in_grad, float* out_grad,
    float lr, float sense_threshold);
void update_z(float* In, float* Out,
    int M, int T, double* z,
    int32_t x, int32_t* context, int64_t context_length,
    int32_t* paths, int8_t* codes, int64_t length);
double init_z(double* pi, int T, double alpha, double d, float min_prob);
""")

with open(os.path.join(os.path.dirname(__file__), 'learn.c'), 'rb') as f:
    superlib = ffi.verify(f.read(), libraries=['m'])


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


init_z = superlib.init_z
inplace_update = superlib.inplace_update
update_z = superlib.update_z

