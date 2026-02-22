#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>
#include <xmmintrin.h>
#include <omp.h>
#include <time.h>

#define INPUT_DIM 512
#define HIDDEN_DIM 2048
#define BIT_TARGET 10000000

// --- 1. STRUCTURES ---
typedef struct {
    float *l1_mu, *l1_sigma, *l1_snake_alpha;
    float *ln1_w, *ln1_b, *l2_mu, *l2_sigma, *l2_snake_alpha;
    float *ln2_w, *ln2_b, *out_w, *out_b;
} Expert;

typedef struct {
    Expert* exp;
    float *state, *temp, *scratch;
    uint64_t seed;
} Stream;

// --- 2. THE SPEED KERNELS ---

// Fast XORShift for Entropy (Thread-Safe)
static inline float mangle_noise(uint64_t* seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    uint32_t irnd = (uint32_t)(*seed & 0x007FFFFF) | 0x3F800000;
    return (*(float*)&irnd) - 1.5f; 
}

// SIMD Sine Approximation
static inline __m256 fast_sin_ps(__m256 x) {
    __m256 pi = _mm256_set1_ps(3.14159265f);
    __m256 inv_pi = _mm256_set1_ps(0.318309f);
    __m256 x_red = _mm256_sub_ps(x, _mm256_mul_ps(pi, _mm256_round_ps(_mm256_mul_ps(x, inv_pi), 0x08)));
    __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x_red);
    return _mm256_mul_ps(x_red, _mm256_sub_ps(_mm256_set1_ps(1.273239f), _mm256_mul_ps(_mm256_set1_ps(0.405284f), abs_x)));
}

void nitro_drift_linear(float* out, float* in, float* mu, float* sigma, int in_dim, int out_dim, float drift, uint64_t* seed) {
    for (int i = 0; i < out_dim; i++) {
        __m256 vnoise = _mm256_set1_ps(mangle_noise(seed) * drift);
        __m256 vsum = _mm256_setzero_ps();
        for (int j = 0; j < in_dim; j += 8) {
            __m256 vmu = _mm256_load_ps(&mu[i * in_dim + j]);
            __m256 vsig = _mm256_load_ps(&sigma[i * in_dim + j]);
            __m256 vin = _mm256_load_ps(&in[j]);
            vsum = _mm256_fmadd_ps(vin, _mm256_fmadd_ps(vsig, vnoise, vmu), vsum);
        }
        __m128 vlow = _mm256_castps256_ps128(vsum);
        vlow = _mm_add_ps(vlow, _mm256_extractf128_ps(vsum, 1));
        vlow = _mm_hadd_ps(vlow, vlow);
        vlow = _mm_hadd_ps(vlow, vlow);
        out[i] = _mm_cvtss_f32(vlow);
    }
}

void nitro_snake(float* x, float* alpha, int dim) {
    for (int i = 0; i < dim; i += 8) {
        __m256 vx = _mm256_load_ps(&x[i]);
        __m256 va = _mm256_load_ps(&alpha[i]);
        __m256 s = fast_sin_ps(_mm256_mul_ps(va, vx));
        _mm256_store_ps(&x[i], _mm256_fmadd_ps(_mm256_div_ps(_mm256_set1_ps(1.0f), va), _mm256_mul_ps(s, s), vx));
    }
}

// --- 3. LOADING LOGIC ---

void load_tensor(float** p, size_t sz, FILE* f) {
    *p = (float*)_mm_malloc(sz * sizeof(float), 32);
    fread(*p, sizeof(float), sz, f);
}

Expert* load_expert(const char* path) {
    FILE* f = fopen(path, "rb");
    if(!f) return NULL;
    Expert* e = malloc(sizeof(Expert));
    load_tensor(&e->l1_mu, 2048*512, f); 
    load_tensor(&e->l1_sigma, 2048*512, f); 
    load_tensor(&e->l1_snake_alpha, 2048, f);
    load_tensor(&e->ln1_w, 2048, f); 
    load_tensor(&e->ln1_b, 2048, f);
    load_tensor(&e->l2_mu, 2048*2048, f); 
    load_tensor(&e->l2_sigma, 2048*2048, f); 
    load_tensor(&e->l2_snake_alpha, 2048, f);
    load_tensor(&e->ln2_w, 2048, f); 
    load_tensor(&e->ln2_b, 2048, f);
    load_tensor(&e->out_w, 2048, f); 
    load_tensor(&e->out_b, 1, f);
    fclose(f);
    return e;
}

// --- 4. EXECUTION ---

float run_nitro_stream(Stream* s) {
    nitro_drift_linear(s->temp, s->state, s->exp->l1_mu, s->exp->l1_sigma, 512, 2048, 0.45f, &s->seed);
    nitro_snake(s->temp, s->exp->l1_snake_alpha, 2048);
    nitro_drift_linear(s->scratch, s->temp, s->exp->l2_mu, s->exp->l2_sigma, 2048, 2048, 0.45f, &s->seed);
    nitro_snake(s->scratch, s->exp->l2_snake_alpha, 2048);
    
    float val = s->exp->out_b[0];
    for(int i=0; i<2048; i++) {
        val += (s->scratch[i] + s->temp[i]) * s->exp->out_w[i];
    }
    memcpy(s->state, s->scratch, 512 * sizeof(float));
    return 1.0f / (1.0f + expf(-val)); 
}

int main() {
    _mm_setcsr(_mm_getcsr() | 0x8040); 
    const char* paths[] = {"SNAKE_BINS/1.bin","SNAKE_BINS/2.bin","SNAKE_BINS/3.bin"};
    Expert* experts[3];
    for(int i=0; i<3; i++) {
        experts[i] = load_expert(paths[i]);
        if(!experts[i]) { printf("Fail: %s\n", paths[i]); return 1; }
    }

    char* bit_buffer = malloc(BIT_TARGET);
    printf("Starting Nitro Engine (Clang-Bug Workaround)...\n");
    double start = omp_get_wtime();
    uint64_t global_base = (uint64_t)time(NULL);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Stream sm[3];
        for(int i=0; i<3; i++) {
            sm[i].exp = experts[i];
            sm[i].state = _mm_malloc(512*4, 32);
            sm[i].temp = _mm_malloc(2048*4, 32);
            sm[i].scratch = _mm_malloc(2048*4, 32);
            
            // Replaced RDRAND with a stable software seed
            sm[i].seed = global_base ^ ((uint64_t)tid << 32) ^ (uint64_t)i;
            for(int j=0; j<512; j++) sm[i].state[j] = mangle_noise(&sm[i].seed);
        }

        #pragma omp for schedule(static)
        for(int i=0; i<BIT_TARGET; i++) {
            uint8_t b = (run_nitro_stream(&sm[0]) > 0.5f) ^ (run_nitro_stream(&sm[1]) > 0.5f) ^ (run_nitro_stream(&sm[2]) > 0.5f);
            bit_buffer[i] = b ? '1' : '0';
        }
    }

    printf("Time: %.2fs\n", omp_get_wtime() - start);
    FILE* out = fopen("generated_bits2.txt", "wb");
    fwrite(bit_buffer, 1, BIT_TARGET, out);
    fclose(out);
    return 0;
}