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
    int32_t x, int32_t* context, int context_length,
    int32_t* paths, int8_t* codes, int64_t length,
    float* in_grad, float* out_grad,
    float lr, float sense_threshold) {

    float pr = 0;
    int32_t y;
    int8_t* code;
    int32_t* path;
    for (int ci = 0; ci < context_length; ci++) {
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


void update_z(float* In, float* Out,
    int M, int T, double* z,
    int32_t x, int32_t* context, int64_t context_length,
    int32_t* paths, int8_t* codes, int64_t length) {

    int32_t y;
    int8_t* code;
    int32_t* path;
    for (int ci = 0; ci < context_length; ci++) {
        y = context[ci];
        path = paths + y * length;
        code = codes + y * length;

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

double mean_beta(double a, double b) {
    return a / (a + b);
}

double meanlog_beta(double a, double b) {
    return digamma(a) - digamma(a + b);
}

double init_z(double* pi, int T, double alpha, double d, float min_prob) {
    double r = 0.;
    double x = 1.;
    double a, b, pi_k;
    int senses = 0;
    int k;

    double ts = 0.;
    for (k = 0; k < T; k++) {
        ts += pi[k];
    }

    for (int k = 0; k < T - 1; k++) {
        ts = fmax(ts - pi[k], 0.);
        a = 1. + pi[k] - d;
        b = alpha + k*d + ts;
        pi[k] = meanlog_beta(a, b) + r;
        r += meanlog_beta(b, a);

        pi_k = mean_beta(a, b) * x;
        x = fmax(x - pi_k, 0.);
        if (pi_k >= min_prob)
            senses += 1;
    }

    pi[T - 1] = r;
    if (x >= min_prob)
        senses += 1;
    return senses;
}
