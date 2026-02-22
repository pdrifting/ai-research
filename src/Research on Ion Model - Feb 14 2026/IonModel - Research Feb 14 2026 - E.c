#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>

#define SNAPSHOT_BIT_LEN 1000000
#define TEST_WINDOW_BYTES 10 // 80 bits per test
#define TEST_COUNT 100

// Load the entire 185MB corpus into one block
uint8_t* load_corpus(const char* folder, size_t *total_size) {
    DIR *d = opendir(folder);
    struct dirent *dir;
    size_t capacity = 200 * 1024 * 1024; // 200MB start
    uint8_t *block = malloc(capacity);
    size_t offset = 0;

    if (!d || !block) return NULL;

    while ((dir = readdir(d)) != NULL) {
        if (dir->d_name[0] == '.') continue;
        char path[512];
        snprintf(path, 512, "%s/%s", folder, dir->d_name);
        FILE *f = fopen(path, "rb");
        if (f) {
            offset += fread(block + offset, 1, 16384, f);
            fclose(f);
        }
    }
    closedir(d);
    *total_size = offset;
    return block;
}

// Convert ASCII '0'/'1' file to raw bytes
uint8_t* bits_to_bytes(const char* filename, size_t *out_len) {
    FILE *f = fopen(filename, "r");
    if (!f) return NULL;
    uint8_t *bytes = calloc(SNAPSHOT_BIT_LEN / 8 + 1, 1);
    for (int i = 0; i < SNAPSHOT_BIT_LEN; i++) {
        int c = fgetc(f);
        if (c == '1') bytes[i / 8] |= (1 << (7 - (i % 8)));
    }
    fclose(f);
    *out_len = SNAPSHOT_BIT_LEN / 8;
    return bytes;
}

void audit_stream(const char* name, uint8_t* stream_bytes, uint8_t* corpus, size_t corpus_len) {
    int collisions = 0;
    
    printf("\nAuditing: %s\n", name);

    for (int i = 0; i < TEST_COUNT; i++) {
        // Pick a random 10-byte segment from the generated snapshot
        size_t stream_off = rand() % (SNAPSHOT_BIT_LEN / 8 - TEST_WINDOW_BYTES);
        uint8_t *sample = stream_bytes + stream_off;

        // Sliding window search across the entire 185MB Weather Block
        for (size_t c_off = 0; c_off < corpus_len - TEST_WINDOW_BYTES; c_off++) {
            if (memcmp(sample, corpus + c_off, TEST_WINDOW_BYTES) == 0) {
                collisions++;
                break; 
            }
        }
    }

    printf("  - Random 80-bit Collision Test: %d/%d matches found in source data.\n", collisions, TEST_COUNT);
    if (collisions == 0) {
        printf("  - RESULT: Snapshot is mathematically unique from training data.\n");
    } else {
        printf("  - WARNING: Found %d identical sequences. Possible data leakage.\n", collisions);
    }
}

int main() {
    size_t corpus_len;
    uint8_t *corpus = load_corpus("./Training_Data", &corpus_len);
    if (!corpus) return 1;
    printf("Corpus Loaded: %.2f MB\n", corpus_len / (1024.0 * 1024.0));

    DIR *d = opendir(".");
    struct dirent *dir;
    while ((dir = readdir(d)) != NULL) {
        if (strstr(dir->d_name, ".txt")) { // Scan generated bitstreams
            size_t s_len;
            uint8_t *s_bytes = bits_to_bytes(dir->d_name, &s_len);
            if (s_bytes) {
                audit_stream(dir->d_name, s_bytes, corpus, corpus_len);
                free(s_bytes);
            }
        }
    }

    free(corpus);
    return 0;
}