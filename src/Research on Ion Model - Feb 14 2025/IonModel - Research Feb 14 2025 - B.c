#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <dirent.h>

#define INPUT_SIZE 16384
#define HIDDEN_WIDTH 1024
#define OUTPUT_SIZE 16384
#define MAX_CONNECTIONS 50

typedef struct {
    int num_indices;
    int indices[MAX_CONNECTIONS];
    float weights[MAX_CONNECTIONS];
    float bias;
} Node;

typedef struct {
    Node nodes[HIDDEN_WIDTH];
} HiddenLayer;

typedef struct {
    Node nodes[OUTPUT_SIZE];
} OutputLayer;

typedef struct {
    HiddenLayer hidden;
    OutputLayer output;
    float learning_rate;
} Model;

// Output still uses Sigmoid to keep results in 0-1 byte range
float sigmoid(float x) {
    if (x > 15.0f) return 0.999999f;
    if (x < -15.0f) return 0.000001f;
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_deriv(float x) { return x * (1.0f - x); }

// Hidden uses Sine to capture high-frequency entropy patterns
float hidden_act(float x) { return sinf(x); }
float hidden_deriv(float x) { return cosf(x); }

void init_node(Node *n, int prev_layer_size) {
    n->num_indices = (rand() % (MAX_CONNECTIONS - 2 + 1)) + 2;
    // Increased scale for Sine activation to ensure we hit multiple periods
    float scale = sqrtf(6.0f / (float)(prev_layer_size + 1)); 
    for (int i = 0; i < n->num_indices; i++) {
        n->indices[i] = rand() % prev_layer_size;
        n->weights[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
    }
    n->bias = (((float)rand() / (float)RAND_MAX) * 2.0f * M_PI); // Random phase
}

void train_step(Model *m, float *input) {
    static float h_out[HIDDEN_WIDTH];
    static float o_out[OUTPUT_SIZE];
    static float o_delta[OUTPUT_SIZE];
    static float h_delta[HIDDEN_WIDTH];

    // Forward with Sine Hidden
    for (int i = 0; i < HIDDEN_WIDTH; i++) {
        float sum = m->hidden.nodes[i].bias;
        for (int j = 0; j < m->hidden.nodes[i].num_indices; j++) 
            sum += input[m->hidden.nodes[i].indices[j]] * m->hidden.nodes[i].weights[j];
        h_out[i] = hidden_act(sum);
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = m->output.nodes[i].bias;
        for (int j = 0; j < m->output.nodes[i].num_indices; j++) 
            sum += h_out[m->output.nodes[i].indices[j]] * m->output.nodes[i].weights[j];
        o_out[i] = sigmoid(sum);
    }

    // Backprop
    for (int i = 0; i < OUTPUT_SIZE; i++) 
        o_delta[i] = (input[i] - o_out[i]) * sigmoid_deriv(o_out[i]);

    memset(h_delta, 0, sizeof(h_delta));
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < m->output.nodes[i].num_indices; j++) 
            h_delta[m->output.nodes[i].indices[j]] += o_delta[i] * m->output.nodes[i].weights[j];
    }
    for (int i = 0; i < HIDDEN_WIDTH; i++) h_delta[i] *= hidden_deriv(h_out[i]);

    // Update with Jitter (helps escape the 0.08 trap)
    float jitter = m->learning_rate * 0.01f;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < m->output.nodes[i].num_indices; j++) {
            float noise = (((float)rand() / (float)RAND_MAX) - 0.5f) * jitter;
            m->output.nodes[i].weights[j] += m->learning_rate * o_delta[i] * h_out[m->output.nodes[i].indices[j]] + noise;
        }
        m->output.nodes[i].bias += m->learning_rate * o_delta[i];
    }
    for (int i = 0; i < HIDDEN_WIDTH; i++) {
        for (int j = 0; j < m->hidden.nodes[i].num_indices; j++) {
            float noise = (((float)rand() / (float)RAND_MAX) - 0.5f) * jitter;
            m->hidden.nodes[i].weights[j] += m->learning_rate * h_delta[i] * input[m->hidden.nodes[i].indices[j]] + noise;
        }
        m->hidden.nodes[i].bias += m->learning_rate * h_delta[i];
    }
}

Model* create_model(float lr) {
    Model *m = (Model*)malloc(sizeof(Model));
    if (!m) return NULL;
    m->learning_rate = lr;
    for (int i = 0; i < HIDDEN_WIDTH; i++) init_node(&m->hidden.nodes[i], INPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; i++) init_node(&m->output.nodes[i], HIDDEN_WIDTH);
    return m;
}

void forward(Model *m, float *input, float *h_out, float *o_out) {
    for (int i = 0; i < HIDDEN_WIDTH; i++) {
        float sum = m->hidden.nodes[i].bias;
        for (int j = 0; j < m->hidden.nodes[i].num_indices; j++) {
            sum += input[m->hidden.nodes[i].indices[j]] * m->hidden.nodes[i].weights[j];
        }
        h_out[i] = sigmoid(sum);
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = m->output.nodes[i].bias;
        for (int j = 0; j < m->output.nodes[i].num_indices; j++) {
            sum += h_out[m->output.nodes[i].indices[j]] * m->output.nodes[i].weights[j];
        }
        o_out[i] = sigmoid(sum);
    }
}

void save_snapshot(Model *m, float loss) {
    char filename[64];
    int loss_int = (int)(loss * 1000000);
    if (loss_int > 999999) loss_int = 999999;
    sprintf(filename, "model_%06d.snap", loss_int);
    
    FILE *f = fopen(filename, "wb");
    if (f) {
        fwrite(m, sizeof(Model), 1, f);
        fclose(f);
        printf("Saved: %s\n", filename);
    }
}

int main() {
    srand((unsigned int)time(NULL));
    Model *m = create_model(0.01f);
    
    float *input = malloc(INPUT_SIZE * sizeof(float));
    unsigned char *raw = malloc(INPUT_SIZE);
    float loss = 1.0f;
    float last_best_loss = 1.0f;
    uint64_t iter = 0;

    printf("Starting training on weather noise...\n");

    while (loss > 0.00001f) {
        DIR *td = opendir("./Training_Data");
        if (!td) {
            printf("Error: ./Training_Data not found!\n");
            return 1;
        }

        struct dirent *entry;
        while ((entry = readdir(td)) != NULL) {
            if (entry->d_name[0] == '.') continue;

            char path[512];
            snprintf(path, 512, "./Training_Data/%s", entry->d_name);
            FILE *f = fopen(path, "rb");
            if (!f) continue;

            if (fread(raw, 1, INPUT_SIZE, f) == INPUT_SIZE) {
                for (int i = 0; i < INPUT_SIZE; i++) {
                    input[i] = (float)raw[i] / 255.0f;
                }
                
                train_step(m, input);

                if (iter % 100 == 0) {
                    static float h_test[HIDDEN_WIDTH], o_test[OUTPUT_SIZE];
                    forward(m, input, h_test, o_test);
                    loss = 0;
                    for (int i = 0; i < OUTPUT_SIZE; i++) {
                        float e = input[i] - o_test[i];
                        loss += e * e;
                    }
                    loss /= OUTPUT_SIZE;

                    printf("Iter: %llu, Loss: %f, LR: %f\n", iter, loss, m->learning_rate);

                    if (loss < last_best_loss * 0.98f) { 
                        save_snapshot(m, loss);
                        last_best_loss = loss;
                    }
                    
                    if (loss < 0.05f) m->learning_rate = 0.005f;
                    if (loss < 0.01f) m->learning_rate = 0.0001f;
                    if (loss < 0.005f) m->learning_rate = 0.0005f;
                    if (loss < 0.001f) m->learning_rate = 0.00001f;
                }
                iter++;
            }
            fclose(f);
        }
        closedir(td);
    }

    free(input);
    free(raw);
    free(m);
    return 0;
}