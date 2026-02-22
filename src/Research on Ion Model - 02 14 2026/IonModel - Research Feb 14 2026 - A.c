#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

#define INPUT_SIZE 16384        // Size of input vector (16 KB of floats)
#define HIDDEN_WIDTH 1024       // Number of hidden nodes
#define OUTPUT_SIZE 16384       // Output matches input (autoencoder)
#define MAX_CONNECTIONS 50      // Each node connects to at most 50 previous-layer nodes

// A sparse node: connects to a limited set of indices with weights + bias
typedef struct {
    int num_indices;                // How many connections this node uses
    int indices[MAX_CONNECTIONS];   // Which input/hidden indices it reads from
    float weights[MAX_CONNECTIONS]; // Weights for each connection
    float bias;                     // Node bias
} Node;

// Hidden layer: 1024 sparse nodes
typedef struct {
    Node nodes[HIDDEN_WIDTH];
} HiddenLayer;

// Output layer: 16384 sparse nodes
typedef struct {
    Node nodes[OUTPUT_SIZE];
} OutputLayer;

// Full model w/sparse hidden layer, sparse output layer, learning rate
typedef struct {
    HiddenLayer hidden;
    OutputLayer output;
    float learning_rate;
} Model;

// Sigmoid activation with clamping to avoid overflow
float sigmoid(float x) {
    if (x > 15.0f) return 0.999999f;
    if (x < -15.0f) return 0.000001f;
    return 1.0f / (1.0f + expf(-x));
}

// Derivative of sigmoid, assuming input is already sigmoid(x)
float sigmoid_deriv(float x) {
    return x * (1.0f - x);
}

// Initialize a sparse node with random connections and weights
void init_node(Node *n, int prev_layer_size) {
    n->num_indices = (rand() % (MAX_CONNECTIONS - 2 + 1)) + 2; // Between 2 and MAX_CONNECTIONS
    float scale = sqrtf(2.0f / (float)(prev_layer_size + 1)); // He-like scaling
    for (int i = 0; i < n->num_indices; i++) {
        n->indices[i] = rand() % prev_layer_size; // Random input index
        n->weights[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale; // Random weight
    }
    n->bias = 0.0f;
}

// Allocate and initialize the model
Model* create_model(float lr) {
    Model *m = (Model*)malloc(sizeof(Model));
    if (!m) return NULL;
    m->learning_rate = lr;

    // Initialize hidden layer nodes
    for (int i = 0; i < HIDDEN_WIDTH; i++) init_node(&m->hidden.nodes[i], INPUT_SIZE);

    // Initialize output layer nodes
    for (int i = 0; i < OUTPUT_SIZE; i++)  init_node(&m->output.nodes[i], HIDDEN_WIDTH);

    return m;
}

// Forward pass through sparse hidden layer and sparse output layer
void forward(Model *m, float *input, float *h_out, float *o_out) {
    // Hidden layer
    for (int i = 0; i < HIDDEN_WIDTH; i++) {
        float sum = m->hidden.nodes[i].bias;
        for (int j = 0; j < m->hidden.nodes[i].num_indices; j++) {
            sum += input[m->hidden.nodes[i].indices[j]] * m->hidden.nodes[i].weights[j];
        }
        h_out[i] = sigmoid(sum);
    }

    // Output layer
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = m->output.nodes[i].bias;
        for (int j = 0; j < m->output.nodes[i].num_indices; j++) {
            sum += h_out[m->output.nodes[i].indices[j]] * m->output.nodes[i].weights[j];
        }
        o_out[i] = sigmoid(sum);
    }
}

// One training step: forward pass, backprop, weight update
void train_step(Model *m, float *input, float *target) {
    static float h_out  [HIDDEN_WIDTH];   // Hidden activations
    static float o_out  [OUTPUT_SIZE];    // Output activations
    static float o_delta[OUTPUT_SIZE];    // Output error gradient
    static float h_delta[HIDDEN_WIDTH];   // Hidden error gradient

    forward(m, input, h_out, o_out);

    // Output layer delta = (target - output) * derivative
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        o_delta[i] = (target[i] - o_out[i]) * sigmoid_deriv(o_out[i]);
    }

    // Backpropagate into hidden layer (sparse)
    memset(h_delta, 0, sizeof(h_delta));
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < m->output.nodes[i].num_indices; j++) {
            h_delta[m->output.nodes[i].indices[j]] += o_delta[i] * m->output.nodes[i].weights[j];
        }
    }

    // Apply derivative to hidden deltas
    for (int i = 0; i < HIDDEN_WIDTH; i++) {
        h_delta[i] *= sigmoid_deriv(h_out[i]);
    }

    // Update output layer weights
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < m->output.nodes[i].num_indices; j++) {
            m->output.nodes[i].weights[j] += m->learning_rate * o_delta[i] * h_out[m->output.nodes[i].indices[j]];
        }
        m->output.nodes[i].bias += m->learning_rate * o_delta[i];
    }

    // Update hidden layer weights
    for (int i = 0; i < HIDDEN_WIDTH; i++) {
        for (int j = 0; j < m->hidden.nodes[i].num_indices; j++) {
            m->hidden.nodes[i].weights[j] += m->learning_rate * h_delta[i] * input[m->hidden.nodes[i].indices[j]];
        }
        m->hidden.nodes[i].bias += m->learning_rate * h_delta[i];
    }
}

// Save model snapshot when loss improves
void save_snapshot(Model *m, float loss) {
    char filename[64];
    int loss_int = (int)(loss * 1000000); // Convert float loss to integer
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
    srand((unsigned int)time(NULL)); // Seed PRNG
    Model *m = create_model(0.01f);  // Create model with LR = 0.01
    
    float *input  = malloc(INPUT_SIZE * sizeof(float));
    float *target = malloc(OUTPUT_SIZE * sizeof(float));
    float loss    = 1.0f;
    uint64_t iter = 0;

    float last_best_loss = 1.0f;

    // Training loop until extremely small loss (never reached in practice)
    while (loss > 0.00001f) {

        // Generate random input and target (autoencoder)
        for (int i = 0; i < INPUT_SIZE; i++) {
            uint8_t r = rand() & 0xFF;     // Random byte
            input[i]  = (float)r / 255.0f; // Normalize
            target[i] = input[i];
        }

        train_step(m, input, target);

        // Every 100 iterations, compute loss and maybe save snapshot
        if (iter % 100 == 0) {
            float h_test[HIDDEN_WIDTH], o_test[OUTPUT_SIZE];
            forward(m, input, h_test, o_test);

            loss = 0;
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                float e = target[i] - o_test[i];
                loss += e * e;
            }
            loss /= OUTPUT_SIZE;

            printf("Iter: %llu, Loss: %f, LR: %f\n", iter, loss, m->learning_rate);

            // Save snapshot if loss improves by 5%
            if (loss < last_best_loss * 0.95f) {
                save_snapshot(m, loss);
                last_best_loss = loss;
            }

            // Adjust learning rate based on loss thresholds
            if (loss < 0.01f)  m->learning_rate = 0.005f;
            if (loss < 0.005f) m->learning_rate = 0.0001f;
            if (loss < 0.001f) m->learning_rate = 0.00005f;
        }
        iter++;
    }

    free(input);
    free(target);
    free(m);

    return 0;
}
