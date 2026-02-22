#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_DIM 2
#define NUM_SAMPLES 4
#define LEARNING_RATE 0.1
#define EPOCHS 1000

typedef struct {
    double weights[INPUT_DIM];
    double bias;
} Perceptron;

double step_function(double z) {
    return z > 0 ? 1.0 : 0.0;
}

void init_perceptron(Perceptron *p) {
    for (int i = 0; i < INPUT_DIM; i++) {
        p->weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    p->bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

double predict(Perceptron *p, double inputs[INPUT_DIM]) {
    double z = 0.0;
    for (int i = 0; i < INPUT_DIM; i++) {
        z += p->weights[i] * inputs[i];
    }
    z += p->bias;
    return step_function(z);
}

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
            printf("Epoch %d, MSE Loss: %f, Weights: [%f, %f], Bias: %f\n",
                   epoch, total_error / NUM_SAMPLES, p->weights[0], p->weights[1], p->bias);
        }
    }
}

void generate_and_data(double X[NUM_SAMPLES][INPUT_DIM], double y[NUM_SAMPLES]) {
    double data[NUM_SAMPLES][INPUT_DIM + 1] = {
        {0, 0, 0},
        {0, 1, 0},
        {1, 0, 0},
        {1, 1, 1}
    };
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            X[i][j] = data[i][j];
        }
        y[i] = data[i][INPUT_DIM];
    }
}

int main() {
    srand(42);
    double X[NUM_SAMPLES][INPUT_DIM];
    double y[NUM_SAMPLES];
    generate_and_data(X, y);

    Perceptron model;
    init_perceptron(&model);
    printf("Initial weights: [%f, %f]\n", model.weights[0], model.weights[1]);
    printf("Initial bias: %f\n", model.bias);
    train(&model, X, y);

    printf("\nFinal weights: [%f, %f]\n", model.weights[0], model.weights[1]);
    printf("Final bias: %f\n\n", model.bias);
    printf("Testing AND gate:\n");
    for (int i = 0; i < NUM_SAMPLES; i++) {
        double prediction = predict(&model, X[i]);
        printf("Input: [%d,%d], Predicted: %d, True: %d\n",
               (int)X[i][0], (int)X[i][1], (int)prediction, (int)y[i]);
    }
    return 0;
}