#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define LATTICE_WIDTH 1024
#define MAX_SYNAPSES 12

typedef enum { XOR_G, NAND_G, OR_G, AND_G, NOR_G } GateType;
typedef struct { uint16_t src; uint8_t dly; uint8_t tmr; uint8_t act; } Synapse;
typedef struct { uint8_t state; GateType type; Synapse dendrites[MAX_SYNAPSES]; } LatticeNeuron;
typedef struct { LatticeNeuron hidden[LATTICE_WIDTH]; } IonModel;

const char* G_NAMES[] = {"XOR", "NAND", "OR", "AND", "NOR"};

void run_circuit_audit(const char* filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { printf("File not found.\n"); return; }
    IonModel *m = malloc(sizeof(IonModel));
    fread(m, sizeof(IonModel), 1, f);
    fclose(f);

    printf("Circuit Analysis of: %s\n", filename);
    printf("====================================================\n");

    // 1. GATE DISTRIBUTION (The Logic Mix)
    int counts[5] = {0};
    for(int i=0; i<LATTICE_WIDTH; i++) counts[m->hidden[i].type]++;
    
    printf("## Logic Species Distribution\n");
    for(int i=0; i<5; i++) printf("%s: %d (%.1f%%)\n", G_NAMES[i], counts[i], (counts[i]/10.24));

    // 2. FEEDBACK NETWORK ANALYSIS (The Netlist)
    int fan_in_recursive[LATTICE_WIDTH] = {0};
    int total_delay = 0;
    
    for (int i = 0; i < LATTICE_WIDTH; i++) {
        for (int j = 0; j < MAX_SYNAPSES; j++) {
            total_delay += m->hidden[i].dendrites[j].dly;
            // The recursive driver is specifically at index 1
            if (j == 1) {
                int source = m->hidden[i].dendrites[j].src % LATTICE_WIDTH;
                fan_in_recursive[source]++;
            }
        }
    }

    // 3. IDENTIFYING "HUB" NEURONS (High Fan-out nodes)
    printf("\n## Top 5 High-Traffic Nodes (Chaos Hubs)\n");
    for(int k=0; k<5; k++) {
        int max_val = -1, hub_idx = -1;
        for(int i=0; i<LATTICE_WIDTH; i++) {
            if(fan_in_recursive[i] > max_val) {
                max_val = fan_in_recursive[i];
                hub_idx = i;
            }
        }
        printf("Neuron %d [%s]: Fan-out to %d neighbors\n", 
                hub_idx, G_NAMES[m->hidden[hub_idx].type], max_val);
        fan_in_recursive[hub_idx] = -1; // Clear for next max find
    }

    // 4. IGNITION PATHWAYS (NAND -> High Delay XOR)
    printf("\n## Critical Ignition Pathways (Zero-State Kickers)\n");
    int chains = 0;
    for (int i = 0; i < LATTICE_WIDTH; i++) {
        if (m->hidden[i].type == NAND_G || m->hidden[i].type == NOR_G) {
            int target_idx = m->hidden[i].dendrites[1].src % LATTICE_WIDTH;
            if (m->hidden[target_idx].type == XOR_G && m->hidden[i].dendrites[1].dly > 12) {
                if (chains < 5) printf("Path: %d [%s] -> %d [XOR] (Prop Delay: %d ticks)\n", 
                                       i, G_NAMES[m->hidden[i].type], target_idx, m->hidden[i].dendrites[1].dly);
                chains++;
            }
        }
    }
    printf("Total Identified High-Delay Ignition Paths: %d\n", chains);

    // 5. THE "SATURATION" DUMP (Why All 1's kills it)
    int destructive_gates = counts[2] + counts[3]; // OR and AND
    printf("\n## Saturation Potential\n");
    printf("Destructive/Static Gates (OR/AND): %d\n", destructive_gates);
    printf("Self-Starting Ratio: %.2f (Higher = more stable on 0s)\n", 
            (float)(counts[1] + counts[4]) / destructive_gates);

    free(m);
}

int main() {
    run_circuit_audit("logic_stress_e7.snap");
    return 0;
}