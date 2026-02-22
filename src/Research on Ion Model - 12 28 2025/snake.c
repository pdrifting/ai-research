#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>

#define INPUT_DIM 512
#define HIDDEN_DIM 2048
#define BIT_TARGET 10000000 // 10 Million bits

typedef struct {
    // L1 DriftLayer
    float *l1_mu, *l1_sigma, *l1_snake_alpha;
    // LN1
    float *ln1_w, *ln1_b;
    // L2 DriftLayer
    float *l2_mu, *l2_sigma, *l2_snake_alpha;
    // LN2
    float *ln2_w, *ln2_b;
    // Out Head
    float *out_w, *out_b;
} ChaosExpert;

ChaosExpert* load_expert(const char* path) {
    ChaosExpert* exp = (ChaosExpert*)malloc(sizeof(ChaosExpert));
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    // Helper to allocate and read
    auto read_tensor = [&](float** ptr, size_t size) {
        *ptr = (float*)_mm_malloc(size * sizeof(float), 32); // 32-byte align for AVX2
        fread(*ptr, sizeof(float), size, f);
    };

    // Ordering must match your Python dump (sorted keys)
    read_tensor(&exp->l1_mu,          2048 * 512);
    read_tensor(&exp->l1_sigma,       2048 * 512);
    read_tensor(&exp->l1_snake_alpha, 2048);
    read_tensor(&exp->l2_mu,          2048 * 2048);
    read_tensor(&exp->l2_sigma,       2048 * 2048);
    read_tensor(&exp->l2_snake_alpha, 2048);
    read_tensor(&exp->ln1_w,          2048);
    read_tensor(&exp->ln1_b,          2048);
    read_tensor(&exp->ln2_w,          2048);
    read_tensor(&exp->ln2_b,          2048);
    read_tensor(&exp->out_w,          1 * 2048);
    read_tensor(&exp->out_b,          1);

    fclose(f);
    return exp;
}

void avx2_snake(float* x, float* alpha, int dim) {
    for (int i = 0; i < dim; i += 8) {
        __m256 vx = _mm256_load_ps(&x[i]);
        __m256 va = _mm256_load_ps(&alpha[i]);
        
        // We can't easily vector-sin without SVML, so we loop or use __m256_sin_ps
        for(int j=0; j<8; j++) {
            float a = alpha[i+j] + 1e-9f;
            float s = sinf(a * x[i+j]);
            x[i+j] = x[i+j] + (1.0f / a) * (s * s);
        }
    }
}

void avx2_drift_linear(float* out, float* in, float* mu, float* sigma, int in_dim, int out_dim, float drift_factor) {
    for (int i = 0; i < out_dim; i++) {
        __m256 vsum = _mm256_setzero_ps();
        for (int j = 0; j < in_dim; j += 8) {
            __m256 vmu = _mm256_load_ps(&mu[i * in_dim + j]);
            __m256 vsig = _mm256_load_ps(&sigma[i * in_dim + j]);
            __m256 vin = _mm256_load_ps(&in[j]);
            
            // Generate jitter: mu + (sigma * rand * 0.3)
            // For max speed, we pre-generate a block of noise 
            // or use a fast xorshift vector
            __m256 vnoise = _mm256_set1_ps(fast_randn() * drift_factor); 
            __m256 vw = _mm256_add_ps(vmu, _mm256_mul_ps(vsig, vnoise));
            
            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vin, vw));
        }
        // Horizontal add of vsum
        float temp[8];
        _mm256_store_ps(temp, vsum);
        out[i] = temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7];
    }
}

typedef struct {
    ChaosExpert* expert;
    float* state;
    float* hidden_temp;
} StreamContext;

// Fast Sigmoid for the out_head
inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// The core manifold step logic
float process_manifold(StreamContext* ctx) {
    // 1. Layer 1: Drift + Snake + LayerNorm
    avx2_drift_linear(ctx->hidden_temp, ctx->state, ctx->expert->l1_mu, 
                      ctx->expert->l1_sigma, INPUT_DIM, HIDDEN_DIM, 0.3f);
    avx2_snake(ctx->hidden_temp, ctx->expert->l1_snake_alpha, HIDDEN_DIM);
    layer_norm(ctx->hidden_temp, ctx->expert->ln1_w, ctx->expert->ln1_b, HIDDEN_DIM, 1e-5f);

    // 2. Layer 2: Drift + Snake + LayerNorm + Residual
    float* l2_out = (float*)_mm_malloc(HIDDEN_DIM * sizeof(float), 32);
    avx2_drift_linear(l2_out, ctx->hidden_temp, ctx->expert->l2_mu, 
                      ctx->expert->l2_sigma, HIDDEN_DIM, HIDDEN_DIM, 0.3f);
    avx2_snake(l2_out, ctx->expert->l2_snake_alpha, HIDDEN_DIM);
    layer_norm(l2_out, ctx->expert->ln2_w, ctx->expert->ln2_b, HIDDEN_DIM, 1e-5f);

    // Residual Connection: x = x + identity (ctx->hidden_temp is identity here)
    for(int i=0; i<HIDDEN_DIM; i++) l2_out[i] += ctx->hidden_temp[i];

    // 3. Out Head: Linear to 1 float
    float final_val = 0.0f;
    for(int i=0; i<HIDDEN_DIM; i++) final_val += l2_out[i] * ctx->expert->out_w[i];
    final_val += ctx->expert->out_b[0];

    // Update state for next iteration (Feedback Loop)
    // We take the first 512 values of l2_out to refresh the input manifold
    memcpy(ctx->state, l2_out, INPUT_DIM * sizeof(float));

    _mm_free(l2_out);
    return fast_sigmoid(final_val);
}

int main() {
    printf("Initializing Chaos Engine on Threadripper 3960X...\n");

    const char* files[3] = {"SNAKE_BINS/1.bin", "SNAKE_BINS/2.bin", "SNAKE_BINS/3.bin"};
    StreamContext streams[3];

    for(int i=0; i<3; i++) {
        streams[i].expert = load_expert(files[i]);
        if(!streams[i].expert) { printf("Failed to load %s\n", files[i]); return 1; }
        
        streams[i].state = (float*)_mm_malloc(INPUT_DIM * sizeof(float), 32);
        streams[i].hidden_temp = (float*)_mm_malloc(HIDDEN_DIM * sizeof(float), 32);
        
        // Seed initial state with some entropy
        for(int j=0; j<INPUT_DIM; j++) streams[i].state[j] = fast_randn();
    }

FILE* out_f = fopen("generated_bits.txt", "w"); // Open as text
    if (!out_f) return 1;

    printf("Generating %d bits as text characters...\n", BIT_TARGET);

    for(int i = 0; i < BIT_TARGET; i++) {
        // Run the 3 manifolds
        float v1 = process_manifold(&streams[0]);
        float v2 = process_manifold(&streams[1]);
        float v3 = process_manifold(&streams[2]);

        // Triple-Mix XOR Junction
        uint8_t b1 = v1 > 0.5f;
        uint8_t b2 = v2 > 0.5f;
        uint8_t b3 = v3 > 0.5f;
        
        uint8_t final_bit = b1 ^ b2 ^ b3;

        // Write as ASCII '0' (48) or '1' (49)
        fputc(final_bit ? '1' : '0', out_f);

        // Optional: Add a newline every 1000 bits to keep the text file readable
        // if (i % 1000 == 999) fputc('\n', out_f);

        if(i % 100000 == 0) {
            printf("Progress: %d%% (%d bits)\r", (i * 100) / BIT_TARGET, i);
            fflush(stdout);
        }
    }

    fclose(out_f);
    printf("\nDone. Output saved to generated_bits.txt\n");

    return 0;
}