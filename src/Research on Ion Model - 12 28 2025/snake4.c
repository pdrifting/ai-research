#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>
#include <xmmintrin.h>
#include <omp.h>

#define INPUT_DIM 512
#define HIDDEN_DIM 2048
#define BIT_TARGET 10000000
#define CHUNK_SIZE 102

// --- 1. CORE UTILITIES ---

static inline float fast_sigmoid(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

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
    float inv_std = 1.0f / sqrtf((var / dim) + 1e-4f);
    for (int i = 0; i < dim; i++) {
        x[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

// --- 2. AVX2 KERNELS ---

void avx2_drift_linear(float* out, float* in, float* mu, float* sigma, int in_dim, int out_dim, float drift) {
    float noise = fast_randn() * drift;
    __m256 vnoise = _mm256_set1_ps(noise);
    for (int i = 0; i < out_dim; i++) {
        __m256 vsum = _mm256_setzero_ps();
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
        if (x[i] > 50.0f) x[i] = 50.0f;
        if (x[i] < -50.0f) x[i] = -50.0f;
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
    float *state, *temp, *scratch;
} Stream;

void load_tensor(float** p, size_t sz, FILE* f) {
    *p = (float*)_mm_malloc(sz * sizeof(float), 32);
    fread(*p, sizeof(float), sz, f);
}

Expert* load_expert(const char* p) {
    FILE* f = fopen(p, "rb");
    if(!f) return NULL;
    Expert* e = malloc(sizeof(Expert));

    // ORDER MATCHING THE VALIDATED PYTHON EXPORT
    load_tensor(&e->l1_mu,          2048 * 512,  f); 
    load_tensor(&e->l1_sigma,       2048 * 512,  f); 
    load_tensor(&e->l1_snake_alpha, 2048,        f); 
    
    load_tensor(&e->ln1_w,          2048,        f); 
    load_tensor(&e->ln1_b,          2048,        f); 

    load_tensor(&e->l2_mu,          2048 * 2048, f); 
    load_tensor(&e->l2_sigma,       2048 * 2048, f); 
    load_tensor(&e->l2_snake_alpha, 2048,        f); 

    load_tensor(&e->ln2_w,          2048,        f); 
    load_tensor(&e->ln2_b,          2048,        f); 

    load_tensor(&e->out_w,          2048,        f); 
    load_tensor(&e->out_b,          1,           f); 

    fclose(f);
    return e;
}

float run_stream(Stream* s) {
    avx2_drift_linear(s->temp, s->state, s->exp->l1_mu, s->exp->l1_sigma, 512, 2048, 0.3f);
    avx2_snake(s->temp, s->exp->l1_snake_alpha, 2048);
    layer_norm(s->temp, s->exp->ln1_w, s->exp->ln1_b, 2048);

    avx2_drift_linear(s->scratch, s->temp, s->exp->l2_mu, s->exp->l2_sigma, 2048, 2048, 0.3f);
    avx2_snake(s->scratch, s->exp->l2_snake_alpha, 2048);
    layer_norm(s->scratch, s->exp->ln2_w, s->exp->ln2_b, 2048);

    float val = s->exp->out_b[0];
    for(int i=0; i<2048; i++) {
        s->scratch[i] += s->temp[i]; 
        val += s->scratch[i] * s->exp->out_w[i];
    }

    if (isnan(val)) {
        for(int j=0; j<512; j++) s->state[j] = fast_randn();
        return 0.5f; 
    }

    memcpy(s->state, s->scratch, 512 * sizeof(float)); 
    return fast_sigmoid(val);
}

int main() {
    _mm_setcsr(_mm_getcsr() | 0x8040); 
    
    int num_threads = omp_get_max_threads();
    printf("Booting SnakeEngine on %d cores...\n", num_threads);

    // 1. Load Experts once (shared across threads)
    const char* paths[] = {"SNAKE_BINS/1.bin","SNAKE_BINS/2.bin","SNAKE_BINS/3.bin"};
    Expert* shared_experts[3];
    for(int i=0; i<3; i++) {
        shared_experts[i] = load_expert(paths[i]);
    }

    // 2. Prepare Output Buffer
    char* final_bits = malloc(BIT_TARGET + 1);
    
    // 3. Parallel Generation Loop
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        // Each thread needs its own local state for the 3 streams
        Stream local_sm[3];
        for(int i=0; i<3; i++) {
            local_sm[i].exp = shared_experts[i];
            local_sm[i].state = _mm_malloc(512 * 4, 32);
            local_sm[i].temp = _mm_malloc(2048 * 4, 32);
            local_sm[i].scratch = _mm_malloc(2048 * 4, 32);
            // Give each thread a unique starting position in the manifold
            for(int j=0; j<512; j++) local_sm[i].state[j] = fast_randn() + tid;
        }

        // Divide work among threads
        #pragma omp for schedule(static)
        for(int i=0; i<BIT_TARGET; i++) {
            uint8_t b = (run_stream(&local_sm[0]) > 0.5f) ^ 
                        (run_stream(&local_sm[1]) > 0.5f) ^ 
                        (run_stream(&local_sm[2]) > 0.5f);
            final_bits[i] = b ? '1' : '0';
            
            if(tid == 0 && i % 1000 == 0) {
                printf("Progress: ~%d bits (Multi-threaded)\r", i * num_threads);
                fflush(stdout);
            }
        }
        
        // Cleanup local thread memory
        for(int i=0; i<3; i++) {
            _mm_free(local_sm[i].state);
            _mm_free(local_sm[i].temp);
            _mm_free(local_sm[i].scratch);
        }
    }

    // 4. Atomic Write to Disk
    FILE* out = fopen("generated_bits.txt", "w");
    fwrite(final_bits, 1, BIT_TARGET, out);
    fclose(out);
    free(final_bits);

    printf("\nDone. Threadripper utilized. File: generated_bits.txt\n");
    return 0;
}