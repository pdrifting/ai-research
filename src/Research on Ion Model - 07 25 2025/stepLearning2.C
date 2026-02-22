#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_DIM 4
#define NUM_SAMPLES 16
#define LEARNING_RATE 0.1
#define EPOCHS 1000

// Perceptron structure
typedef struct {
    double weights[INPUT_DIM];
    double bias;
} Perceptron;

// Step function
double step_function(double z) {
    return z > 0 ? 1.0 : 0.0;
}

// Initialize perceptron with random weights
void init_perceptron(Perceptron *p) {
    for (int i = 0; i < INPUT_DIM; i++) {
        p->weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Random between -1 and 1
    }
    p->bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

// Predict function
double predict(Perceptron *p, double inputs[INPUT_DIM]) {
    double z = 0.0;
    for (int i = 0; i < INPUT_DIM; i++) {
        z += p->weights[i] * inputs[i];
    }
    z += p->bias;
    return step_function(z);
}

// Train perceptron
void train(Perceptron *p, double X[NUM_SAMPLES][INPUT_DIM], double y[NUM_SAMPLES]) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_error = 0.0;
        for (int i = 0; i < NUM_SAMPLES; i++) {
            double prediction = predict(p, X[i]);
            double error = y[i] - prediction;
            total_error += error * error;

            if (error != 0) {
                for (int j = 0; j < INPUT_DIM; j++) {
                    p->weights[j] += LEARNING_RATE * error * X[i][j];
                }
                p->bias += LEARNING_RATE * error;
            }
        }
        if (epoch % 100 == 0) {
            printf("Epoch %d, MSE Loss: %f, Weights: [%f, %f, %f, %f], Bias: %f\n",
                   epoch, total_error / NUM_SAMPLES,
                   p->weights[0], p->weights[1], p->weights[2], p->weights[3], p->bias);
        }
    }
}

// Generate training data
void generate_sign_data(double X[NUM_SAMPLES][INPUT_DIM], double y[NUM_SAMPLES]) {
    for (int i = -8; i < 8; i++) {
        int idx = i + 8;
        for (int j = 0; j < INPUT_DIM; j++) {
            X[idx][j] = (i >> j) & 1;
        }
        y[idx] = i < 0 ? 1.0 : 0.0;
    }
}

int main() {
    // Set random seed
    srand(42);

    // Training data
    double X[NUM_SAMPLES][INPUT_DIM];
    double y[NUM_SAMPLES];
    generate_sign_data(X, y);

    // Initialize and train perceptron
    Perceptron model;
    init_perceptron(&model);
    printf("Initial weights: [%f, %f, %f, %f]\n", model.weights[0], model.weights[1], model.weights[2], model.weights[3]);
    printf("Initial bias: %f\n", model.bias);
    train(&model, X, y);

    // Test the model
    printf("\nFinal weights: [%f, %f, %f, %f]\n", model.weights[0], model.weights[1], model.weights[2], model.weights[3]);
    printf("Final bias: %f\n\n", model.bias);
    printf("Testing sign extraction:\n");
    for (int i = 0; i < NUM_SAMPLES; i++) {
        double prediction = predict(&model, X[i]);
        printf("Input: [%d,%d,%d,%d] (decimal: %d), Predicted sign: %d, True sign: %d\n",
               (int)X[i][0], (int)X[i][1], (int)X[i][2], (int)X[i][3], i-8, (int)prediction, (int)y[i]);
    }
    return 0;
}