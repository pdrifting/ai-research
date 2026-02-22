#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#define LATTICE_WIDTH 1024
#define INPUT_SIZE 16384
#define MAX_SYNAPSES 12
#define TEST_BIT_LEN 1000000

typedef enum { XOR_G, NAND_G, OR_G, AND_G, NOR_G } GateType;
typedef struct { uint16_t src; uint8_t dly; uint8_t tmr; uint8_t act; } Synapse;
typedef struct { uint8_t state; GateType type; Synapse dendrites[MAX_SYNAPSES]; } LatticeNeuron;
typedef struct { LatticeNeuron hidden[LATTICE_WIDTH]; } IonModel;

static uint8_t output_bits[LATTICE_WIDTH];
static uint8_t input_zeros[INPUT_SIZE] = {0};

// --- PHYSICS ENGINE ---
void tick_physics(IonModel *m, uint8_t *in, uint8_t *out) {
    for (int i = 0; i < LATTICE_WIDTH; i++) {
        LatticeNeuron *n = &m->hidden[i];
        // Input A from external buffer, Input B from lattice recursion
        uint8_t a = in[n->dendrites[0].src % INPUT_SIZE];
        uint8_t b = m->hidden[n->dendrites[1].src % LATTICE_WIDTH].state;
        
        uint8_t g_out = 0;
        switch(n->type) {
            case XOR_G:  g_out = a ^ b; break;
            case NAND_G: g_out = !(a && b); break;
            case OR_G:   g_out = a || b; break;
            case AND_G:  g_out = a && b; break;
            case NOR_G:  g_out = !(a || b); break;
        }

        uint8_t flux = 0;
        for (int j = 0; j < MAX_SYNAPSES; j++) {
            Synapse *s = &n->dendrites[j];
            if (g_out) s->act = 1;
            if (s->act) {
                if (s->tmr >= s->dly) { 
                    flux ^= 1; 
                    s->tmr = 0; 
                    s->act = 0; 
                } else {
                    s->tmr++;
                }
            }
        }
        n->state ^= (flux & 1);
        out[i] = n->state;
    }
}

// --- HIGH-GAIN TOPOLOGY SEEDING ---
void seed_snapshot7_v3_damping(IonModel *m) {
    GateType recipe[LATTICE_WIDTH];
    int i = 0;
    // Shifted for Damping: Less NAND/NOR, More AND/OR (The "Brakes")
    for (; i < 300; i++) recipe[i] = XOR_G;   // Entropy mixing
    for (; i < 450; i++) recipe[i] = NAND_G;  // Startup torque (Reduced)
    for (; i < 600; i++) recipe[i] = NOR_G;   // Startup torque (Reduced)
    for (; i < 850; i++) recipe[i] = OR_G;    // Signal Sink
    for (; i < 1024; i++) recipe[i] = AND_G;  // Signal Sink

    for (int j = 0; j < LATTICE_WIDTH; j++) {
        int r = rand() % LATTICE_WIDTH;
        GateType tmp = recipe[j]; recipe[j] = recipe[r]; recipe[r] = tmp;
    }

    for (int n = 0; n < LATTICE_WIDTH; n++) {
        m->hidden[n].type = recipe[n];
        m->hidden[n].state = rand() % 2;
        for (int s = 0; s < MAX_SYNAPSES; s++) {
            m->hidden[n].dendrites[s].src = (n + (rand() % 128) - 64 + LATTICE_WIDTH) % LATTICE_WIDTH;
            
            // INCREASE DELAY SPREAD: Longer delays lower the frequency/density
            // Using larger prime numbers to decorrelate the loops
            int slow_primes[] = {13, 17, 19, 23, 29}; 
            if (s == 1) 
                m->hidden[n].dendrites[s].dly = slow_primes[rand() % 5];
            else
                m->hidden[n].dendrites[s].dly = 8 + (rand() % 16); // Minimum 8 tick delay
                
            m->hidden[n].dendrites[s].tmr = 0;
            m->hidden[n].dendrites[s].act = 0;
        }
    }
}

// --- METRIC MEASUREMENT ---
float measure_0_density(IonModel *m) {
    uint8_t out[LATTICE_WIDTH], prev[LATTICE_WIDTH];
    uint32_t flips = 0;
    memcpy(prev, output_bits, LATTICE_WIDTH);
    
    for (int t = 0; t < 200; t++) {
        tick_physics(m, input_zeros, out);
        for (int k = 0; k < LATTICE_WIDTH; k++) {
            if (out[k] != prev[k]) flips++;
        }
        memcpy(prev, out, LATTICE_WIDTH);
    }
    return (float)flips / (200 * LATTICE_WIDTH);
}

void inject_jitter(IonModel *m) {
    for (int i = 0; i < 50; i++) {
        int target = rand() % LATTICE_WIDTH;
        int syn = rand() % MAX_SYNAPSES;
        // Shift delay by +/- 1 tick to knock phase out of alignment
        if (m->hidden[target].dendrites[syn].dly > 2) {
            m->hidden[target].dendrites[syn].dly += (rand() % 3) - 1;
        }
    }
}

int main() {
    srand(time(NULL));
    IonModel *m = calloc(1, sizeof(IonModel));
    seed_snapshot7_v3_damping(m); // Start with our best guess

    float last_d0 = 0;
    int stability_count = 0;

    printf("Hunting the Snapshot 7 Attractor...\n");

    for (int step = 0; step < 1000000; step++) {
        tick_physics(m, input_zeros, output_bits);

        if (step % 250 == 0) {
            float d0 = measure_0_density(m);
            printf("Step %d | Density: %.4f ", step, d0);

            // Check for "Dead Orbits" (Static Density)
            if (fabs(d0 - last_d0) < 0.0005) {
                stability_count++;
                printf("[Stable: %d] ", stability_count);
            } else {
                stability_count = 0;
            }
            last_d0 = d0;

            // TRIGGER: Goldilocks Zone
            if (d0 >= 0.235 && d0 <= 0.265) {
                printf("\n>>> CANDIDATE DISCOVERED <<<\n");
                // [Save Logic Here]
                break; 
            }

            // ACTION: If too stable and too low, inject jitter to "spark" it
            if (stability_count > 4) {
                if (d0 < 0.23) {
                    printf("-> Injecting Jitter (Low Gain)");
                    inject_jitter(m); 
                } else if (d0 > 0.27) {
                    printf("-> Cooling (High Gain)");
                    seed_snapshot7_v3_damping(m); // Too hot, use the brakes
                }
                stability_count = 0;
            }
            printf("\n");
        }
    }
    free(m);
    return 0;
}