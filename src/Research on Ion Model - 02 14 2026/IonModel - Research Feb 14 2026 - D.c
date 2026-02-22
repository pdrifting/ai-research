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

static uint8_t input_spikes[INPUT_SIZE];
static uint8_t output_bits[LATTICE_WIDTH];

void process_xor_snapshot(const char *snap_path, IonModel *m) {
    FILE *sf = fopen(snap_path, "rb");
    if (!sf) return;
    fread(m, sizeof(IonModel), 1, sf);
    fclose(sf);

    char out_name[256];
    strncpy(out_name, snap_path, 250);
    char *dot = strrchr(out_name, '.');
    if (dot) *dot = '\0';
    strcat(out_name, ".txt");

    FILE *out_f = fopen(out_name, "w");
    if (!out_f) return;

    DIR *td = opendir("./Training_Data");
    if (!td) { fclose(out_f); return; }

    uint8_t *raw_data = malloc(INPUT_SIZE);
    int bits_written = 0;
    struct dirent *entry;

    printf("Processing %s -> %s\n", snap_path, out_name);

    while ((entry = readdir(td)) != NULL && bits_written < TOTAL_BITS) {
        if (entry->d_name[0] == '.') continue;
        
        char path[512];
        snprintf(path, 512, "./Training_Data/%s", entry->d_name);
        FILE *df = fopen(path, "rb");
        if (!df) continue;

        if (fread(raw_data, 1, INPUT_SIZE, df) == INPUT_SIZE) {
            for (int i = 0; i < INPUT_SIZE; i++) input_spikes[i] = (raw_data[i] > 127);

            for (int t = 0; t < SIM_TICKS && bits_written < TOTAL_BITS; t++) {
                // PHYSICS TICK (Matching Trainer)
                for (int i = 0; i < LATTICE_WIDTH; i++) {
                    LatticeNeuron *n = &m->hidden[i];
                    uint8_t flux = 0;
                    for (int j = 0; j < MAX_SYNAPSES; j++) {
                        Synapse *s = &n->dendrites[j];
                        if (input_spikes[s->source_idx]) s->signal_active = 1;
                        if (s->signal_active) {
                            if (s->timer >= s->delay) {
                                flux ^= 1;
                                s->timer = 0; s->signal_active = 0;
                            } else s->timer++;
                        }
                    }
                    n->state ^= flux;
                    output_bits[i] = n->state;
                }

                // BIT EXTRACTION: XOR-Summing 4 distant neurons for high entropy
                for (int i = 0; i < 125 && bits_written < TOTAL_BITS; i++) {
                    uint8_t b = output_bits[i] ^ 
                                output_bits[i + 256] ^ 
                                output_bits[i + 512] ^ 
                                output_bits[i + 768];
                    
                    fputc(b ? '1' : '0', out_f);
                    bits_written++;
                }
            }
        }
        fclose(df);
    }

    closedir(td);
    fclose(out_f);
    free(raw_data);
}

int main() {
    IonModel *m = malloc(sizeof(IonModel));
    DIR *d = opendir(".");
    struct dirent *dir;
    while ((dir = readdir(d)) != NULL) {
        if (strstr(dir->d_name, ".snap")) process_xor_snapshot(dir->d_name, m);
    }
    closedir(d);
    free(m);
    return 0;
}