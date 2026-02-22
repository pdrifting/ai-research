#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>

#define INPUT_DIM 512
#define HIDDEN_DIM 2048
#define BIT_TARGET 10000000

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// --- 1. CORE MATH UTILITIES ---

// Fast PRNG for weight drift (Box-Muller)
float fast_randn() {
    static uint64_t seed = 0x123456789ABCDEF0ULL;
    seed = seed * 6364136223846793005ULL + 1;
    float u1 = (float)(seed >> 40) / 16777216.0f;
    seed = seed * 6364136223846793005ULL + 1;
    float u2 = (float)(seed >> 40) / 16777216.0f;
    return sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(2.0f * (float)M_PI * u2);
}

void layer_norm(float* x, float* weight, float* bias, int dim, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= dim;

    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var = sqrtf((var / dim) + eps);

    for (int i = 0; i < dim; i++) {
        x[i] = ((x[i] - mean) / var) * weight[i] + bias[i];
    }
}

// --- 2. AVX2 KERNELS ---

void avx2_snake(float* x, float* alpha, int dim) {
    for (int i = 0; i < dim; i += 8) {
        // x + (1.0 / alpha) * (sin(alpha * x)^2)
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
        float noise = fast_randn() * drift_factor;
        __m256 vnoise = _mm256_set1_ps(noise);

        for (int j = 0; j < in_dim; j += 8) {
            __m256 vmu = _mm256_load_ps(&mu[i * in_dim + j]);
            __m256 vsig = _mm256_load_ps(&sigma[i * in_dim + j]);
            __m256 vin = _mm256_load_ps(&in[j]);
            
            // w = mu + (sigma * noise)
            __m256 vw = _mm256_add_ps(vmu, _mm256_mul_ps(vsig, vnoise));
            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vin, vw));
        }
        
        float temp[8];
        _mm256_store_ps(temp, vsum);
        out[i] = temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7];
    }
}

// --- 3. INFRASTRUCTURE ---

typedef struct {
    float *l1_mu, *l1_sigma, *l1_snake_alpha;
    float *ln1_w, *ln1_b;
    float *l2_mu, *l2_sigma, *l2_snake_alpha;
    float *ln2_w, *ln2_b;
    float *out_w, *out_b;
} ChaosExpert;

typedef struct {
    ChaosExpert* expert;
    float* state;
    float* hidden_temp;
} StreamContext;

// Helper function replacing the lambda
void read_tensor_helper(float** ptr, size_t size, FILE* f) {
    *ptr = (float*)_mm_malloc(size * sizeof(float), 32);
    if (fread(*ptr, sizeof(float), size, f) != size) {
        fprintf(stderr, "Error reading tensor data\n");
    }
}

ChaosExpert* load_expert(const char* path) {
    ChaosExpert* exp = (ChaosExpert*)malloc(sizeof(ChaosExpert));
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    read_tensor_helper(&exp->l1_mu,          2048 * 512, f);
    read_tensor_helper(&exp->l1_sigma,       2048 * 512, f);
    read_tensor_helper(&exp->l1_snake_alpha, 2048, f);
    read_tensor_helper(&exp->l2_mu,          2048 * 2048, f);
    read_tensor_helper(&exp->l2_sigma,       2048 * 2048, f);
    read_tensor_helper(&exp->l2_snake_alpha, 2048, f);
    read_tensor_helper(&exp->ln1_w,          2048, f);
    read_tensor_helper(&exp->ln1_b,          2048, f);
    read_tensor_helper(&exp->ln2_w,          2048, f);
    read_tensor_helper(&exp->ln2_b,          2048, f);
    read_tensor_helper(&exp->out_w,          2048, f);
    read_tensor_helper(&exp->out_b,          1, f);

    fclose(f);
    return exp;
}

float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float process_manifold(StreamContext* ctx) {
    // Layer 1
    avx2_drift_linear(ctx->hidden_temp, ctx->state, ctx->expert->l1_mu, 
                      ctx->expert->l1_sigma, INPUT_DIM, HIDDEN_DIM, 0.3f);
    avx2_snake(ctx->hidden_temp, ctx->expert->l1_snake_alpha, HIDDEN_DIM);
    layer_norm(ctx->hidden_temp, ctx->expert->ln1_w, ctx->expert->ln1_b, HIDDEN_DIM, 1e-5f);

    // Layer 2
    float* l2_out = (float*)_mm_malloc(HIDDEN_DIM * sizeof(float), 32);
    avx2_drift_linear(l2_out, ctx->hidden_temp, ctx->expert->l2_mu, 
                      ctx->expert->l2_sigma, HIDDEN_DIM, HIDDEN_DIM, 0.3f);
    avx2_snake(l2_out, ctx->expert->l2_snake_alpha, HIDDEN_DIM);
    layer_norm(l2_out, ctx->expert->ln2_w, ctx->expert->ln2_b, HIDDEN_DIM, 1e-5f);

    // Residual + Out
    float final_val = 0.0f;
    for(int i=0; i<HIDDEN_DIM; i++) {
        l2_out[i] += ctx->hidden_temp[i];
        final_val += l2_out[i] * ctx->expert->out_w[i];
    }
    final_val += ctx->expert->out_b[0];

    memcpy(ctx->state, l2_out, INPUT_DIM * sizeof(float));
    _mm_free(l2_out);

    return fast_sigmoid(final_val);
}

// --- 4. MAIN ---

int main() {
    printf("Initializing Chaos Engine (AVX2/Zen2 Optimized)...\n");

    const char* files[3] = {"SNAKE_BINS/1.bin", "SNAKE_BINS/2.bin", "SNAKE_BINS/3.bin"};
    StreamContext streams[3];

    for(int i=0; i<3; i++) {
        streams[i].expert = load_expert(files[i]);
        if(!streams[i].expert) { printf("Failed to load %s\n", files[i]); return 1; }
        streams[i].state = (float*)_mm_malloc(INPUT_DIM * sizeof(float), 32);
        streams[i].hidden_temp = (float*)_mm_malloc(HIDDEN_DIM * sizeof(float), 32);
        for(int j=0; j<INPUT_DIM; j++) streams[i].state[j] = fast_randn();
    }

    FILE* out_f = fopen("generated_bits.txt", "w");
    if (!out_f) return 1;

    // Buffer the file output for performance
    char file_buf[1048576];
    setvbuf(out_f, file_buf, _IOFBF, sizeof(file_buf));

    for(int i = 0; i < BIT_TARGET; i++) {
        float v1 = process_manifold(&streams[0]);
        float v2 = process_manifold(&streams[1]);
        float v3 = process_manifold(&streams[2]);

        uint8_t final_bit = (v1 > 0.5f) ^ (v2 > 0.5f) ^ (v3 > 0.5f);
        fputc(final_bit ? '1' : '0', out_f);

        if(i % 100000 == 0) {
            printf("Progress: %d%% (%d bits)\r", (i * 100) / BIT_TARGET, i);
            fflush(stdout);
        }
    }

    fclose(out_f);
    printf("\nSuccess. 10M bits emitted to generated_bits.txt\n");
    return 0;
}