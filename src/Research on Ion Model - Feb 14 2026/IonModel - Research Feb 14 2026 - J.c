#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define LATTICE_WIDTH 1024
#define MAX_SYNAPSES 12

typedef enum { XOR_G, NAND_G, OR_G, AND_G, NOR_G } GateType;
typedef struct { uint16_t src; uint8_t dly; uint8_t tmr; uint8_t act; } Synapse;
typedef struct { uint8_t state; GateType type; Synapse dendrites[MAX_SYNAPSES]; } LatticeNeuron;
typedef struct { LatticeNeuron hidden[LATTICE_WIDTH]; } IonModel;

const char* G_NAMES[] = {"XOR", "NAND", "OR", "AND", "NOR"};

void trace_ignition_rings(IonModel *m) {
    printf("Ignition Ring Netlist (Snapshot 7)\n");
    printf("==================================\n");
    printf("Source | Gate | Target | Gate | Path Delay | Feedback Type\n");

    for (int i = 0; i < LATTICE_WIDTH; i++) {
        // We only care about gates that can flip a 0 (NAND/NOR)
        if (m->hidden[i].type == NAND_G || m->hidden[i].type == NOR_G) {
            
            // Check the primary recursive synapse (index 1)
            uint16_t target_idx = m->hidden[i].dendrites[1].src % LATTICE_WIDTH;
            uint8_t delay = m->hidden[i].dendrites[1].dly;
            
            // Look for "Perfect Rings": Target feeds back to Source
            uint16_t back_idx = m->hidden[target_idx].dendrites[1].src % LATTICE_WIDTH;
            uint8_t back_delay = m->hidden[target_idx].dendrites[1].dly;
            
            if (back_idx == i) {
                printf("%6d | %4s | %6d | %4s | %10d | DIRECT LOOP\n", 
                       i, G_NAMES[m->hidden[i].type], 
                       target_idx, G_NAMES[m->hidden[target_idx].type], 
                       delay + back_delay);
            } else if (delay > 14) {
                // Look for "Long Propagations" even if not a direct loop
                printf("%6d | %4s | %6d | %4s | %10d | DEEP PROPAGATION\n", 
                       i, G_NAMES[m->hidden[i].type], 
                       target_idx, G_NAMES[m->hidden[target_idx].type], 
                       delay);
            }
        }
    }
}

int main() {
    FILE *f = fopen("logic_stress_e7.snap", "rb");
    if (!f) return 1;
    IonModel *m = malloc(sizeof(IonModel));
    fread(m, sizeof(IonModel), 1, f);
    fclose(f);

    trace_ignition_rings(m);
    free(m);
    return 0;
}