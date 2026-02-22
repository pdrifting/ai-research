#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>

#define INPUT_DIM 512
#define HIDDEN_DIM 2048
#define BIT_TARGET 10000000

// --- 1. CORE UTILITIES ---

static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Box-Muller for weight drift jitter
float fast_randn() {
    static uint64_t seed = 0x123456789ABCDEF0ULL;
    seed = seed * 6364136223846793005ULL + 1;
    float u1 = (float)(seed >> 40) / 16777216.0f;
    seed = seed * 6364136223846793005ULL + 1;
    float u2 = (float)(seed >> 40) / 16777216.0f;
    return sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(2.0f * 3.14159f * u2);
}

void layer_norm(float* x, float* weight, float* bias, int dim) {
    float mean = 0, var = 0;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= dim;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float inv_std = 1.0f / sqrtf((var / dim) + 1e-5f);
    for (int i = 0; i < dim; i++) {
        x[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

// --- 2. AVX2 KERNELS ---

void avx2_drift_linear(float* out, float* in, float* mu, float* sigma, int in_dim, int out_dim, float drift) {
    for (int i = 0; i < out_dim; i++) {
        __m256 vsum = _mm256_setzero_ps();
        __m256 vnoise = _mm256_set1_ps(fast_randn() * drift);
        for (int j = 0; j < in_dim; j += 8) {
            __m256 vmu = _mm256_load_ps(&mu[i * in_dim + j]);
            __m256 vsig = _mm256_load_ps(&sigma[i * in_dim + j]);
            __m256 vin = _mm256_load_ps(&in[j]);
            __m256 vw = _mm256_add_ps(vmu, _mm256_mul_ps(vsig, vnoise));
            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vin, vw));
        }
        float t[8];
        _mm256_store_ps(t, vsum);
        out[i] = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
    }
}

void avx2_snake(float* x, float* alpha, int dim) {
    for (int i = 0; i < dim; i++) {
        // Safety Clamp: Prevents manifold explosion (NaN/Inf)
        if (x[i] > 100.0f) x[i] = 100.0f;
        if (x[i] < -100.0f) x[i] = -100.0f;

        float a = alpha[i] + 1e-9f;
        float s = sinf(a * x[i]);
        x[i] += (1.0f / a) * (s * s);
    }
}

// --- 3. EXECUTION ENGINE ---

typedef struct {
    float *l1_mu, *l1_sigma, *l1_snake_alpha;
    float *ln1_w, *ln1_b, *l2_mu, *l2_sigma, *l2_snake_alpha;
    float *ln2_w, *ln2_b, *out_w, *out_b;
} Expert;

typedef struct {
    Expert* exp;
    float *state, *temp;
} Stream;

void load_tensor(float** p, size_t sz, FILE* f) {
    *p = (float*)_mm_malloc(sz * sizeof(float), 32);
    fread(*p, sizeof(float), sz, f);
}

Expert* load_expert(const char* p) {
    FILE* f = fopen(p, "rb");
    if(!f) return NULL;
    Expert* e = malloc(sizeof(Expert));
    load_tensor(&e->l1_mu, 2048*512, f); load_tensor(&e->l1_sigma, 2048*512, f);
    load_tensor(&e->l1_snake_alpha, 2048, f); load_tensor(&e->l2_mu, 2048*2048, f);
    load_tensor(&e->l2_sigma, 2048*2048, f); load_tensor(&e->l2_snake_alpha, 2048, f);
    load_tensor(&e->ln1_w, 2048, f); load_tensor(&e->ln1_b, 2048, f);
    load_tensor(&e->ln2_w, 2048, f); load_tensor(&e->ln2_b, 2048, f);
    load_tensor(&e->out_w, 2048, f); load_tensor(&e->out_b, 1, f);
    fclose(f);
    return e;
}

float run_stream(Stream* s) {
    // CHANGE: Use a local buffer instead of static to be thread-safe 
    // and avoid any memory fencing delays.
    float scratch[2048] __attribute__((aligned(32))); 
    
    avx2_drift_linear(s->temp, s->state, s->exp->l1_mu, s->exp->l1_sigma, 512, 2048, 0.3f);
    avx2_snake(s->temp, s->exp->l1_snake_alpha, 2048);
    layer_norm(s->temp, s->exp->ln1_w, s->exp->ln1_b, 2048);

    avx2_drift_linear(scratch, s->temp, s->exp->l2_mu, s->exp->l2_sigma, 2048, 2048, 0.3f);
    avx2_snake(scratch, s->exp->l2_snake_alpha, 2048);
    layer_norm(scratch, s->exp->ln2_w, s->exp->ln2_b, 2048);

    float val = s->exp->out_b[0];
    for(int i=0; i<2048; i++) {
        scratch[i] += s->temp[i]; 
        val += scratch[i] * s->exp->out_w[i];
    }

    // DEBUG: Check for NaN
    if (isnan(val)) {
        // If it fails, re-seed the state
        for(int j=0; j<512; j++) s->state[j] = fast_randn();
        return 0.5f; 
    }

    memcpy(s->state, scratch, 512 * sizeof(float)); 
    return fast_sigmoid(val);
}

int main() {
    printf("Booting SnakeEngine on Threadripper...\n"); fflush(stdout);
    
    Stream sm[3];
    const char* paths[] = {"SNAKE_BINS/1.bin","SNAKE_BINS/2.bin","SNAKE_BINS/3.bin"};
    
    for(int i=0; i<3; i++) {
        sm[i].exp = load_expert(paths[i]);
        if(!sm[i].exp) { printf("Error loading %s\n", paths[i]); return 1; }
        sm[i].state = _mm_malloc(512*4, 32);
        sm[i].temp = _mm_malloc(2048*4, 32);
        for(int j=0; j<512; j++) sm[i].state[j] = fast_randn();
    }

    FILE* out = fopen("generated_bits.txt", "w");
    for(int i=0; i<BIT_TARGET; i++) {
        uint8_t b = (run_stream(&sm[0]) > 0.5f) ^ (run_stream(&sm[1]) > 0.5f) ^ (run_stream(&sm[2]) > 0.5f);
        fputc(b ? '1' : '0', out);
        if(i % 100000 == 0) { printf("Progress: %d%%\r", i/100000); fflush(stdout); }
    }
    
    fclose(out);
    printf("\nDone. File: generated_bits.txt\n");
    return 0;
}