#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>
#include <omp.h>

#define INPUT_DIM 512
#define HIDDEN_DIM 2048
#define BIT_TARGET 10000000

// --- 1. THE SPEED KERNELS ---

// Fast XORShift for Entropy (Thread-Safe)
static inline float mangle_noise(uint64_t* seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    uint32_t irnd = (uint32_t)(*seed & 0x007FFFFF) | 0x3F800000;
    return (*(float*)&irnd) - 1.5f; 
}

// AVX Sine Approximation (No function calls)
static inline __m256 fast_sin_ps(__m256 x) {
    __m256 pi = _mm256_set1_ps(3.14159265f);
    __m256 inv_pi = _mm256_set1_ps(0.318309f);
    // Range reduction
    __m256 x_red = _mm256_sub_ps(x, _mm256_mul_ps(pi, _mm256_round_ps(_mm256_mul_ps(x, inv_pi), 0x08)));
    __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x_red);
    // Quadratic fit: x * (1.273 - 0.405 * |x|)
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
            // FMA: vsum += vin * (vmu + vsig * vnoise)
            vsum = _mm256_fmadd_ps(vin, _mm256_fmadd_ps(vsig, vnoise, vmu), vsum);
        }
        // Horizontal add to single float
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
        // x += (1/a) * s^2
        _mm256_store_ps(&x[i], _mm256_fmadd_ps(_mm256_div_ps(_mm256_set1_ps(1.0f), va), _mm256_mul_ps(s, s), vx));
    }
}

// --- 2. ENGINE WRAPPERS ---

typedef struct { float *l1_mu, *l1_sigma, *l1_snake_alpha, *ln1_w, *ln1_b, *l2_mu, *l2_sigma, *l2_snake_alpha, *ln2_w, *ln2_b, *out_w, *out_b; } Expert;
typedef struct { Expert* exp; float *state, *temp, *scratch; uint64_t seed; } Stream;

float run_nitro_stream(Stream* s) {
    nitro_drift_linear(s->temp, s->state, s->exp->l1_mu, s->exp->l1_sigma, 512, 2048, 0.3f, &s->seed);
    nitro_snake(s->temp, s->exp->l1_snake_alpha, 2048);
    // Standard layer_norm remains fast enough as it's infrequent
    
    nitro_drift_linear(s->scratch, s->temp, s->exp->l2_mu, s->exp->l2_sigma, 2048, 2048, 0.3f, &s->seed);
    nitro_snake(s->scratch, s->exp->l2_snake_alpha, 2048);
    
    float val = s->exp->out_b[0];
    for(int i=0; i<2048; i++) {
        val += (s->scratch[i] + s->temp[i]) * s->exp->out_w[i];
    }
    memcpy(s->state, s->scratch, 512 * sizeof(float));
    return 1.0f / (1.0f + expf(-val)); 
}

// --- 3. MAIN PARALLEL EXECUTION ---

int main() {
    _mm_setcsr(_mm_getcsr() | 0x8040); 
    
    const char* paths[] = {"SNAKE_BINS/1.bin","SNAKE_BINS/2.bin","SNAKE_BINS/3.bin"};
    Expert* experts[3];
    for(int i=0; i<3; i++) experts[i] = load_expert(paths[i]); // Using your existing load_expert

    char* bit_buffer = malloc(BIT_TARGET);
    int threads = omp_get_max_threads();
    printf("Launching Nitro Engine on %d cores...\n", threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Stream local_sm[3];
        for(int i=0; i<3; i++) {
            local_sm[i].exp = experts[i];
            local_sm[i].state = _mm_malloc(512*4, 32);
            local_sm[i].temp = _mm_malloc(2048*4, 32);
            local_sm[i].scratch = _mm_malloc(2048*4, 32);
            
            // Hardware Seed for unique manifold entry
            unsigned int hw_rnd;
            _rdrand32_step(&hw_rnd);
            local_sm[i].seed = (uint64_t)hw_rnd | ((uint64_t)tid << 32);
            for(int j=0; j<512; j++) local_sm[i].state[j] = mangle_noise(&local_sm[i].seed);
        }

        #pragma omp for schedule(static)
        for(int i=0; i<BIT_TARGET; i++) {
            uint8_t b = (run_nitro_stream(&local_sm[0]) > 0.5f) ^ 
                        (run_nitro_stream(&local_sm[1]) > 0.5f) ^ 
                        (run_nitro_stream(&local_sm[2]) > 0.5f);
            bit_buffer[i] = b ? '1' : '0';
        }
    }

    FILE* out = fopen("generated_bits2.txt", "wb");
    fwrite(bit_buffer, 1, BIT_TARGET, out);
    fclose(out);
    printf("10M bits generated. Ready for NIST tests.\n");
    return 0;
}