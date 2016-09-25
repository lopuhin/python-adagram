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


float inplace_update(
        float* In, float* Out,
        int M, int T, double* z,
        int32_t x, int32_t* context, int context_length,
        int32_t* paths, int8_t* codes, int64_t length,
        float* in_grad, float* out_grad,
        float lr, float sense_threshold) {

    float pr = 0;
    int32_t y;
    int8_t* code;
    int32_t* path;
    for (int ci = 0; ci < context_length; ++ci) {
        y = context[ci];
        path = paths + y * length;
        code = codes + y * length;

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
    }

    return pr;
}


void update_z(
        float* In, float* Out,
        int M, int T, double* z,
        int32_t x, int32_t* context, int64_t context_length,
        int32_t* paths, int8_t* codes, int64_t length) {

    int32_t y;
    int8_t* code;
    int32_t* path;
    for (int ci = 0; ci < context_length; ++ci) {
        y = context[ci];
        path = paths + y * length;
        code = codes + y * length;

        for (int n = 0; n < length && code[n] != -1; ++n) {
            float* out = Out + path[n] * M;

            for (int k = 0; k < T; ++k) {
                float* in = in_offset(In, x, k, M, T);

                float f = 0;
                for (int i = 0; i < M; ++i)
                    f += in[i] * out[i];

                z[k] += logsigmoid(f * (1 - 2*code[n]));
            }
        }
    }

    double z_max = z[0], z_k;
    for (int k = 1; k < T; ++k) {
        z_k = z[k];
        if (z_k > z_max) {
            z_max = z_k;
        }
    }
    double z_sum = 0.;
    for (int k = 0; k < T; ++k) {
        z_k = exp(z[k] - z_max);
        z[k] = z_k;
        z_sum += z_k;
    }
    for (int k = 0; k < T; ++k) {
        z[k] /= z_sum;
    }
}

