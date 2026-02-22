#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <time.h>

#define INPUT_SIZE 16384
#define LATTICE_WIDTH 1024
#define MAX_SYNAPSES 12
#define MAX_DELAY 16 
#define SIM_TICKS 8    
#define EPOCHS 15

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

// Allocating these as globals moves them out of the stack entirely
static uint8_t input_spikes[INPUT_SIZE];
static uint8_t output_bits[LATTICE_WIDTH];

void init_model(IonModel *m) {
    for (int i = 0; i < LATTICE_WIDTH; i++) {
        m->hidden[i].state = rand() % 2;
        m->hidden[i].type = (GateType)(rand() % 5);
        for (int j = 0; j < MAX_SYNAPSES; j++) {
            m->hidden[i].dendrites[j].source_idx = rand() % INPUT_SIZE;
            m->hidden[i].dendrites[j].delay = (rand() % MAX_DELAY) + 1;
            m->hidden[i].dendrites[j].timer = 0;
            m->hidden[i].dendrites[j].signal_active = 0;
        }
    }
}

void tick_logic_physics(IonModel *m, uint8_t *in, uint8_t *out) {
    for (int i = 0; i < LATTICE_WIDTH; i++) {
        LatticeNeuron *n = &m->hidden[i];
        
        // Safety: Explicitly bound the neighbor index
        uint16_t idx_a = n->dendrites[0].source_idx; // External
        uint16_t idx_b = n->dendrites[1].source_idx % LATTICE_WIDTH; // Internal
        
        uint8_t a = in[idx_a];
        uint8_t b = m->hidden[idx_b].state;
        
        uint8_t gate_out = 0;
        switch(n->type) {
            case XOR_G:  gate_out = a ^ b; break;
            case NAND_G: gate_out = !(a && b); break;
            case OR_G:   gate_out = a || b; break;
            case AND_G:  gate_out = a && b; break;
            case NOR_G:  gate_out = !(a || b); break;
        }

        uint8_t flux = 0;
        for (int j = 0; j < MAX_SYNAPSES; j++) {
            Synapse *s = &n->dendrites[j];
            if (gate_out) s->signal_active = 1;

            if (s->signal_active) {
                if (s->timer >= s->delay) {
                    flux ^= 1;
                    s->timer = 0;
                    s->signal_active = 0;
                } else {
                    s->timer++;
                }
            }
        }
        n->state ^= (flux & 1);
        out[i] = n->state;
    }
}

int main() {
    srand(time(NULL));
    
    // Allocate model on Heap
    IonModel *m = (IonModel *)calloc(1, sizeof(IonModel));
    if (!m) { fprintf(stderr, "Heap Allocation Failed\n"); return 1; }
    
    init_model(m);

    DIR *d = opendir("./Training_Data");
    if (!d) { fprintf(stderr, "Training_Data folder missing\n"); free(m); return 1; }

    char (*file_list)[256] = calloc(4000, 256);
    int file_count = 0;
    struct dirent *dir;
    while ((dir = readdir(d)) != NULL && file_count < 4000) {
        if (dir->d_name[0] != '.') {
            strncpy(file_list[file_count++], dir->d_name, 255);
        }
    }
    closedir(d);

    uint8_t *weather_data = malloc(INPUT_SIZE);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        printf("\n--- Logic Stress Epoch %d ---\n", epoch + 1);

        for (int f_idx = 0; f_idx < file_count; f_idx++) {
            char path[512];
            snprintf(path, 512, "./Training_Data/%s", file_list[f_idx]);
            FILE *f = fopen(path, "rb");
            if (f) {
                if(fread(weather_data, 1, INPUT_SIZE, f) == INPUT_SIZE) {
                    for (int i = 0; i < INPUT_SIZE; i++) input_spikes[i] = (weather_data[i] > 127);
                    
                    // Train Sequence
                    for (int t = 0; t < SIM_TICKS; t++) tick_logic_physics(m, input_spikes, output_bits);
                    
                    // Stress 0s
                    memset(input_spikes, 0, INPUT_SIZE);
                    for (int t = 0; t < SIM_TICKS; t++) tick_logic_physics(m, input_spikes, output_bits);
                    
                    // Stress 1s
                    memset(input_spikes, 1, INPUT_SIZE);
                    for (int t = 0; t < SIM_TICKS; t++) tick_logic_physics(m, input_spikes, output_bits);
                }
                fclose(f);
            }
            if (f_idx % 500 == 0) printf("File %d/%d processed\n", f_idx, file_count);
        }

        char fn[64]; sprintf(fn, "logic_stress_e%d.snap", epoch + 1);
        FILE *sf = fopen(fn, "wb");
        if (sf) {
            fwrite(m, sizeof(IonModel), 1, sf);
            fclose(sf);
            printf("Saved %s\n", fn);
        }
    }

    free(weather_data);
    free(file_list);
    free(m);
    return 0;
}