#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dirent.h>

#define LATTICE_WIDTH 1024
#define MAX_SYNAPSES 12
#define INPUT_SIZE 16384

typedef enum { XOR_G, NAND_G, OR_G, AND_G, NOR_G } GateType;

typedef struct {
    uint16_t source_idx;
    uint8_t delay;
    uint8_t timer;
    uint8_t signal_active;
} Synapse;

typedef struct {
    uint8_t state;
    GateType type; 
    Synapse dendrites[MAX_SYNAPSES];
} LatticeNeuron;

typedef struct {
    LatticeNeuron hidden[LATTICE_WIDTH];
} IonModel;

void analyze_snapshot(const char *filename, IonModel *m) {
    FILE *f = fopen(filename, "rb");
    if (!f) return;
    
    // Check file size to ensure it matches current struct
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (size != sizeof(IonModel)) {
        printf("Skipping %s (Size mismatch: %ld vs %ld)\n", filename, size, sizeof(IonModel));
        fclose(f);
        return;
    }

    fread(m, sizeof(IonModel), 1, f);
    fclose(f);

    uint32_t gate_counts[5] = {0};
    uint32_t total_delay = 0;
    uint32_t total_recursive = 0;
    uint32_t self_loops = 0; // Synapses looking back at the same neuron

    for (int i = 0; i < LATTICE_WIDTH; i++) {
        gate_counts[m->hidden[i].type]++;
        
        for (int j = 0; j < MAX_SYNAPSES; j++) {
            Synapse *s = &m->hidden[i].dendrites[j];
            total_delay += s->delay;
            
            // In our recursive logic: idx_b = source_idx % LATTICE_WIDTH
            uint16_t target = s->source_idx % LATTICE_WIDTH;
            if (target == i) self_loops++;
            
            // We consider the first two dendrites the "primary drivers" of the gate
            if (j < 2) total_recursive++; 
        }
    }

    float avg_delay = (float)total_delay / (LATTICE_WIDTH * MAX_SYNAPSES);
    // Ignition potential is the % of gates that flip 0s into 1s (NAND and NOR)
    float ignition = ((float)(gate_counts[1] + gate_counts[4]) / LATTICE_WIDTH) * 100.0f;

    printf("\n--- Analysis for %s ---\n", filename);
    printf("Gate Mix: [XOR:%d] [NAND:%d] [OR:%d] [AND:%d] [NOR:%d]\n", 
           gate_counts[0], gate_counts[1], gate_counts[2], gate_counts[3], gate_counts[4]);
    printf("Ignition Potential: %.2f%% (NAND+NOR)\n", ignition);
    printf("Mean Synaptic Delay: %.3f ticks\n", avg_delay);
    printf("Structural Chaos:   %u self-loops found\n", self_loops);
    
    // Entropy Density Prediction
    // High XOR + NAND/NOR usually equals better NIST scores
    float health = (gate_counts[0] + gate_counts[1] + gate_counts[4]) / 10.24f;
    printf("Lattice Health Score: %.2f/100\n", health);
}

int main() {
    IonModel *m = malloc(sizeof(IonModel));
    if (!m) return 1;

    DIR *d = opendir(".");
    struct dirent *dir;
    
    printf("Lattice Structural Auditor starting...\n");
    printf("========================================\n");

    while ((dir = readdir(d)) != NULL) {
        if (strstr(dir->d_name, ".snap")) {
            analyze_snapshot(dir->d_name, m);
        }
    }

    closedir(d);
    free(m);
    return 0;
}