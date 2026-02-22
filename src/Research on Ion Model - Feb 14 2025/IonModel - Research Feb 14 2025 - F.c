#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dirent.h>

#define INPUT_SIZE 16384
#define LATTICE_WIDTH 1024
#define MAX_SYNAPSES 12 
#define MAX_DELAY 16 
#define SIM_TICKS 8 
#define TOTAL_BITS 1000000

typedef struct {
    uint16_t source_idx;
    uint8_t delay;
    uint8_t timer;
    uint8_t signal_active;
} Synapse;

typedef struct {
    uint8_t state;
    Synapse dendrites[MAX_SYNAPSES];
} LatticeNeuron;

typedef struct {
    LatticeNeuron hidden[LATTICE_WIDTH];
} IonModel;

// The "Autonomous" Physics: Feedback-driven chaos
void tick_recursive_physics(IonModel *m, uint8_t *in, uint8_t *out) {
    for (int i = 0; i < LATTICE_WIDTH; i++) {
        LatticeNeuron *n = &m->hidden[i];
        uint8_t flux = 0;

        for (int j = 0; j < MAX_SYNAPSES; j++) {
            Synapse *s = &n->dendrites[j];
            
            /* RECURSIVE COUPLING:
               Instead of just reading 'in[s->source_idx]', we XOR it with 
               a 'neighbor' neuron's state. This creates a feedback loop 
               where the lattice's own history drives its future.
            */
            uint8_t neighbor_idx = (i + s->source_idx) % LATTICE_WIDTH;
            uint8_t trigger = in[s->source_idx] ^ m->hidden[neighbor_idx].state;

            if (trigger) s->signal_active = 1;

            if (s->signal_active) {
                if (s->timer >= s->delay) {
                    flux ^= 1;
                    s->timer = 0;
                    s->signal_active = 0;
                } else s->timer++;
            }
        }
        n->state ^= flux;
        out[i] = n->state;
    }
}

void run_autonomous_zerotest(const char *snap_path, IonModel *m) {
    FILE *sf = fopen(snap_path, "rb");
    if (!sf) return;
    fread(m, sizeof(IonModel), 1, sf);
    fclose(sf);

    char out_name[256];
    snprintf(out_name, 256, "AUTO_ZERO_%s.txt", snap_path);
    FILE *out_f = fopen(out_name, "w");
    
    uint8_t dead_input[INPUT_SIZE] = {0}; // Strictly all zeros
    uint8_t local_output[LATTICE_WIDTH];
    int bits_written = 0;

    printf("Executing Autonomous Zero-Test: %s\n", snap_path);

    while (bits_written < TOTAL_BITS) {
        for (int t = 0; t < SIM_TICKS && bits_written < TOTAL_BITS; t++) {
            
            tick_recursive_physics(m, dead_input, local_output);

            // Extract via 4-way XOR Whitening
            for (int i = 0; i < 125 && bits_written < TOTAL_BITS; i++) {
                uint8_t b = local_output[i] ^ 
                            local_output[i + 250] ^ 
                            local_output[i + 500] ^ 
                            local_output[i + 750];
                fputc(b ? '1' : '0', out_f);
                bits_written++;
            }
        }
    }
    fclose(out_f);
}

int main() {
    IonModel *m = malloc(sizeof(IonModel));
    DIR *d = opendir(".");
    struct dirent *dir;
    while ((dir = readdir(d)) != NULL) {
        if (strstr(dir->d_name, ".snap")) {
            run_autonomous_zerotest(dir->d_name, m);
        }
    }
    closedir(d);
    free(m);
    return 0;
}