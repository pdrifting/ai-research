#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <time.h>

#define INPUT_SIZE 16384
#define LATTICE_WIDTH 1024
#define MAX_SYNAPSES 12   // Balanced for diffusion without saturation
#define MAX_DELAY 16 
#define SIM_TICKS 8    
#define EPOCHS 10

typedef struct {
    uint16_t source_idx;
    uint8_t delay;
    uint8_t timer;
    uint8_t signal_active;
} Synapse;

typedef struct {
    uint8_t state;        // Current binary state (0 or 1)
    Synapse dendrites[MAX_SYNAPSES];
} LatticeNeuron;

typedef struct {
    LatticeNeuron hidden[LATTICE_WIDTH];
} IonModel;

char files[4000][256];
int file_count = 0;

void init_model(IonModel *m) {
    for (int i = 0; i < LATTICE_WIDTH; i++) {
        m->hidden[i].state = rand() % 2;
        for (int j = 0; j < MAX_SYNAPSES; j++) {
            m->hidden[i].dendrites[j].source_idx = rand() % INPUT_SIZE;
            m->hidden[i].dendrites[j].delay = rand() % MAX_DELAY;
            m->hidden[i].dendrites[j].timer = 0;
            m->hidden[i].dendrites[j].signal_active = 0;
        }
    }
}

// Logic: Neuron state XORs with delayed input pulses
void tick_xor_physics(IonModel *m, uint8_t *in, uint8_t *out) {
    for (int i = 0; i < LATTICE_WIDTH; i++) {
        LatticeNeuron *n = &m->hidden[i];
        uint8_t incoming_flux = 0;

        for (int j = 0; j < MAX_SYNAPSES; j++) {
            Synapse *s = &n->dendrites[j];
            
            // If input is high, prime the synapse
            if (in[s->source_idx]) s->signal_active = 1;

            if (s->signal_active) {
                if (s->timer >= s->delay) {
                    incoming_flux ^= 1; // Pulse arrives, flip the bit
                    s->timer = 0;
                    s->signal_active = 0;
                } else {
                    s->timer++;
                }
            }
        }
        // State update: Current state XOR Flux
        n->state ^= incoming_flux;
        out[i] = n->state;
    }
}

// Evolution: If a neuron becomes static (too many 0s or 1s), rewire a synapse
void evolve_topology(IonModel *m, int neuron_idx) {
    int syn_idx = rand() % MAX_SYNAPSES;
    m->hidden[neuron_idx].dendrites[syn_idx].source_idx = rand() % INPUT_SIZE;
    m->hidden[neuron_idx].dendrites[syn_idx].delay = rand() % MAX_DELAY;
}

void shuffle_files() {
    for (int i = file_count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        char temp[256];
        strcpy(temp, files[i]);
        strcpy(files[i], files[j]);
        strcpy(files[j], temp);
    }
}

int main() {
    srand(time(NULL));
    IonModel *m = malloc(sizeof(IonModel));
    init_model(m);

    DIR *d = opendir("./Training_Data");
    struct dirent *dir;
    if (!d) return 1;
    while ((dir = readdir(d)) != NULL && file_count < 4000) {
        if (dir->d_name[0] != '.') strcpy(files[file_count++], dir->d_name);
    }
    closedir(d);

    uint8_t *raw_data = malloc(INPUT_SIZE);
    uint8_t input_spikes[INPUT_SIZE];
    uint8_t output_bits[LATTICE_WIDTH];

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        shuffle_files();
        printf("\n--- Epoch %d ---\n", epoch + 1);

        for (int f_idx = 0; f_idx < file_count; f_idx++) {
            char path[512];
            snprintf(path, 512, "./Training_Data/%s", files[f_idx]);
            FILE *f = fopen(path, "rb");
            if (!f) continue;
            fread(raw_data, 1, INPUT_SIZE, f);
            fclose(f);

            for (int i = 0; i < INPUT_SIZE; i++) input_spikes[i] = (raw_data[i] > 127);

            uint32_t active_sum = 0;
            for (int t = 0; t < SIM_TICKS; t++) {
                tick_xor_physics(m, input_spikes, output_bits);
                for(int i=0; i<LATTICE_WIDTH; i++) active_sum += output_bits[i];
            }

            // Monitor activity: Ideal is 50% (LATTICE_WIDTH * SIM_TICKS / 2)
            if (f_idx % 500 == 0) {
                printf("File %d, Activity Density: %.2f%%\n", f_idx, (active_sum * 100.0) / (LATTICE_WIDTH * SIM_TICKS));
                
                // Adaptive Rewiring: If a neuron is "dead" or "stuck", evolve it
                if (active_sum < 1000 || active_sum > 7000) {
                     evolve_topology(m, rand() % LATTICE_WIDTH);
                }
            }
        }

        char fn[64]; sprintf(fn, "xor_lattice_e%d.snap", epoch + 1);
        FILE *sf = fopen(fn, "wb");
        fwrite(m, sizeof(IonModel), 1, sf);
        fclose(sf);
    }

    free(raw_data); free(m);
    return 0;
}