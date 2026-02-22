// Neural Memory Graph Simulation with Conceptual Abstraction and Reinforcement
// Adds structured, level-based logging (ERROR, INFO, DEBUG)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>

#ifndef LOG_LEVEL
  #define LOG_LEVEL 2   // 0=ERROR, 1=INFO, 2=DEBUG
#endif

static FILE* __log_file = NULL;

// Return “HH:MM:SS” timestamp
static const char* log_timestamp() {
    static char buf[9];
    time_t t = time(NULL);
    struct tm* lt = localtime(&t);
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d",
             lt->tm_hour, lt->tm_min, lt->tm_sec);
    return buf;
}

static void log_msg(const char* level, const char* fmt, va_list args) {
    FILE* out = __log_file ? __log_file : stdout;
    fprintf(out, "[%s] %-5s: ", log_timestamp(), level);
    vfprintf(out, fmt, args);
    fprintf(out, "\n");
    fflush(out);
}

static void log_error(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    log_msg("ERROR", fmt, ap);
    va_end(ap);
}
#if LOG_LEVEL >= 1
static void log_info(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    log_msg("INFO", fmt, ap);
    va_end(ap);
}
#else
  #define log_info(fmt, ...) ((void)0)
#endif

#if LOG_LEVEL >= 2
static void log_debug(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    log_msg("DEBUG", fmt, ap);
    va_end(ap);
}
#else
  #define log_debug(fmt, ...) ((void)0)
#endif

#define MAX_CONNECTIONS    16
#define MAX_DATA_SIZE      64
#define SIMULATION_STEPS   10
#define SIMILARITY_THRESHOLD 0.7f
#define REDUNDANCY_THRESHOLD 0.95f
#define REINFORCEMENT_BONUS  0.05f
#define CONFIDENCE_GOAL      0.99f

typedef enum { NODE_EXACT, NODE_CONCEPT } NodeType;

typedef struct {
    int    id;
    int*   data;
    int*   lastData;
    int    numConnections;
    int*   connections;
    float  frequency;
    float  similarityScore;
    float  confidenceChange;
    float  confidence;
    float  expansionThreshold;
    NodeType type;
} Node;

typedef struct {
    Node* nodes;
    int   numNodes;
    int   nextAvailableNodeId;
    int   maxConnections;
} IonModel;

static float constrain(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

static int* stringToIntegerArray(const char* str, int size) {
    int* arr = malloc(sizeof(int) * (size + 1));
    for (int i = 0; i < size; i++) arr[i] = (int)str[i];
    arr[size] = -1;
    return arr;
}

static void freeIntegerArray(int* arr) {
    free(arr);
}

static float calculateSimilarity(int* a, int* b) {
    int i = 0;
    double dot = 0, magA = 0, magB = 0;
    while (a[i] != -1 && b[i] != -1) {
        dot  += a[i] * b[i];
        magA += a[i] * a[i];
        magB += b[i] * b[i];
        i++;
    }
    if (magA == 0 || magB == 0) return 0.0f;
    return (float)(dot / (sqrt(magA) * sqrt(magB)));
}

static int isRedundant(IonModel* ionModel, int* candidateData) {
    for (int i = 0; i < ionModel->numNodes; i++) {
        float sim = calculateSimilarity(candidateData, ionModel->nodes[i].data);
        if (sim > REDUNDANCY_THRESHOLD) {
            log_debug("isRedundant: candidate ~ node %d (sim=%.2f)", ionModel->nodes[i].id, sim);
            return 1;
        }
    }
    return 0;
}

static void mutateData(char* buffer, size_t size) {
    if (size < 1) return;
    int index = rand() % size;
    buffer[index] ^= 0x1;
    log_debug("mutateData: flipped bit at pos %d -> '%c'", index, buffer[index]);
}

static void updateMetrics(Node* node) {
    node->frequency += 1.0f;
    if (node->lastData) {
        node->similarityScore = calculateSimilarity(node->data, node->lastData);
    }
    node->confidenceChange = node->similarityScore;
    node->confidence += node->confidenceChange;
    node->confidence = constrain(node->confidence, 0.0f, 1.0f);

    if (node->lastData) free(node->lastData);
    int len = 0;
    while (node->data[len] != -1) len++;
    node->lastData = malloc(sizeof(int) * (len + 1));
    for (int i = 0; i <= len; i++) node->lastData[i] = node->data[i];

    log_debug("updateMetrics: node %d freq=%.0f sim=%.2f Δconf=%.2f conf=%.2f",
              node->id, node->frequency, node->similarityScore,
              node->confidenceChange, node->confidence);
}

static void connectNodes(IonModel* ionModel, int id1, int id2) {
    Node* n1 = &ionModel->nodes[id1];
    Node* n2 = &ionModel->nodes[id2];
    float sim = calculateSimilarity(n1->data, n2->data);
    log_debug("connectNodes: %d↔%d sim=%.2f", n1->id, n2->id, sim);

    if (sim > SIMILARITY_THRESHOLD &&
        n1->numConnections < MAX_CONNECTIONS &&
        n2->numConnections < MAX_CONNECTIONS) {
        n1->connections[n1->numConnections++] = n2->id;
        n2->connections[n2->numConnections++] = n1->id;
        log_info("Linked node %d -> %d (sim=%.2f)", id1, id2, sim);
        updateMetrics(n1);
        updateMetrics(n2);
    }
}

static void pruneNetwork(IonModel* ionModel) {
    log_info("Pruning network...");
    for (int i = 0; i < ionModel->numNodes; i++) {
        Node* node = &ionModel->nodes[i];
        node->confidence *= 0.98f;
        node->confidence = constrain(node->confidence, 0.0f, 1.0f);
        for (int j = 0; j < node->numConnections;) {
            int connId = node->connections[j];
            if (ionModel->nodes[connId].confidence < 0.3f) {
                log_debug("Prune conn %d -/-> %d (conf=%.2f)",
                          node->id, connId, ionModel->nodes[connId].confidence);
                for (int k = j; k < node->numConnections - 1; k++)
                    node->connections[k] = node->connections[k + 1];
                node->numConnections--;
            } else {
                j++;
            }
        }
    }
}

static void expandNetwork(IonModel* ionModel, const char* rawData, int preserveExact) {
    char buffer[256];
    strncpy(buffer, rawData, 255);
    buffer[255] = '\0';
    log_debug("expandNetwork: raw='%s' preserveExact=%d", rawData, preserveExact);

    if (!preserveExact) mutateData(buffer, strlen(buffer));

    int* candidate = stringToIntegerArray(buffer, strlen(buffer));
    if (!preserveExact && isRedundant(ionModel, candidate)) {
        log_info("expandNetwork: redundant '%s', skipping", buffer);
        free(candidate);
        return;
    }

    ionModel->nodes = realloc(ionModel->nodes,
                           sizeof(Node) * (ionModel->numNodes + 1));
    Node* newNode = &ionModel->nodes[ionModel->numNodes];
    newNode->id = ionModel->nextAvailableNodeId++;
    newNode->data = candidate;
    newNode->lastData = NULL;
    newNode->connections = malloc(sizeof(int) * MAX_CONNECTIONS);
    newNode->numConnections = 0;
    newNode->frequency = 0;
    newNode->similarityScore = 0;
    newNode->confidence = 0.5f;
    newNode->confidenceChange = 0;
    newNode->expansionThreshold = 0.5f;
    newNode->type = preserveExact ? NODE_EXACT : NODE_CONCEPT;

    log_info("Created node %d [%s] '%s'",
             newNode->id,
             preserveExact ? "EXACT" : "CONCEPT",
             buffer);

    for (int i = 0; i < ionModel->numNodes; i++) {
        connectNodes(ionModel, i, ionModel->numNodes);
    }
    ionModel->numNodes++;
    updateMetrics(newNode);
}

static void reinforceNode(Node* node) {
    node->confidence += REINFORCEMENT_BONUS;
    node->confidence = constrain(node->confidence, 0.0f, 1.0f);
    log_debug("reinforceNode: node %d new conf=%.2f",
              node->id, node->confidence);
}

static void reinforceHighConfidence(IonModel* ionModel) {
    for (int i = 0; i < ionModel->numNodes; i++) {
        if (ionModel->nodes[i].confidence > 0.8f) {
            log_info("reinforceHighConfidence: boosting node %d", ionModel->nodes[i].id);
            reinforceNode(&ionModel->nodes[i]);
        }
    }
}

static int reachedConfidenceGoal(IonModel* ionModel) {
    if (ionModel->numNodes == 0) {
        log_debug("reachedConfidenceGoal: no nodes yet, returning false");
        return 0;
    }
    for (int i = 0; i < ionModel->numNodes; i++) {
        if (ionModel->nodes[i].confidence < CONFIDENCE_GOAL) {
            log_debug("reachedConfidenceGoal: node %d conf=%.2f < %.2f",
                      ionModel->nodes[i].id,
                      ionModel->nodes[i].confidence,
                      CONFIDENCE_GOAL);
            return 0;
        }
    }
    return 1;
}

static void generateConceptNode(IonModel* ionModel, int idxA, int idxB) {
    // copy out the IDs *before* realloc
    int idA = ionModel->nodes[idxA].id;
    int idB = ionModel->nodes[idxB].id;

    // measure lengths
    int lenA = 0, lenB = 0;
    while (ionModel->nodes[idxA].data[lenA] != -1) lenA++;
    while (ionModel->nodes[idxB].data[lenB] != -1) lenB++;
    int minLen = lenA < lenB ? lenA : lenB;

    // build merged vector
    int* merged = malloc(sizeof(int) * (minLen + 1));
    for (int i = 0; i < minLen; i++) {
        merged[i] = (ionModel->nodes[idxA].data[i] == ionModel->nodes[idxB].data[i])
                    ? ionModel->nodes[idxA].data[i]
                    : (ionModel->nodes[idxA].data[i] + ionModel->nodes[idxB].data[i]) / 2;
    }
    merged[minLen] = -1;

    // now resize the array
    Node* newSpace = realloc(ionModel->nodes,
                             sizeof(Node) * (ionModel->numNodes + 1));
    if (!newSpace) {
        log_error("generateConceptNode: realloc failed!");
        free(merged);
        return;
    }
    ionModel->nodes = newSpace;

    // initialize the new concept node
    Node* concept = &ionModel->nodes[ionModel->numNodes];
    concept->id                = ionModel->nextAvailableNodeId++;
    concept->data              = merged;
    concept->lastData          = NULL;
    concept->connections       = malloc(sizeof(int) * MAX_CONNECTIONS);
    concept->numConnections    = 0;
    concept->frequency         = 0;
    concept->similarityScore   = 0;
    concept->confidence        = 0.3f;
    concept->confidenceChange  = 0;
    concept->expansionThreshold= 0.5f;
    concept->type              = NODE_CONCEPT;

    // connect back to parents
    concept->connections[concept->numConnections++] = idA;
    concept->connections[concept->numConnections++] = idB;
    ionModel->numNodes++;

    log_info("generateConceptNode: created concept %d from %d & %d",
             concept->id, idA, idB);

    updateMetrics(concept);
}


static void debugNetwork(IonModel* ionModel, int step) {
    printf("\n--- STEP %d ---\n", step);
    for (int i = 0; i < ionModel->numNodes; i++) {
        Node* node = &ionModel->nodes[i];
        const char* label = node->type == NODE_EXACT ? "EXACT" : "CONCEPT";
        printf("Node %d [%s] | Conf: %.2f | Conns:",
               node->id, label, node->confidence);
        for (int j = 0; j < node->numConnections; j++)
            printf(" %d", node->connections[j]);
        printf("\n");
    }
}

static void simulateTextGrowth(IonModel* ionModel, const char* textCorpus[], int numEntries) {
    log_info("Starting text growth simulation for %d entries", numEntries);
    int step = 0;
    while (step < SIMULATION_STEPS && !reachedConfidenceGoal(ionModel)) {
        const char* data = textCorpus[rand() % numEntries];
        log_info("Step %d: input='%s'", step, data);

        expandNetwork(ionModel, data, 1);
        expandNetwork(ionModel, data, 0);

        if (ionModel->numNodes > 1) {
            /* pass indices, not raw pointers */
            int idxA = ionModel->numNodes - 2;
            int idxB = ionModel->numNodes - 1;
            generateConceptNode(ionModel, idxA, idxB);
        }

        pruneNetwork(ionModel);
        reinforceHighConfidence(ionModel);

        debugNetwork(ionModel, step);
        step++;
    }
    log_info("Simulation ended at step %d", step);
}

int main() {
    srand((unsigned int)time(NULL));

    __log_file = fopen("simulation.log", "w");
    if (!__log_file) {
        log_error("Cannot open simulation.log, logging to stdout");
    }

    IonModel ionModel = { 0 };
    ionModel.maxConnections = MAX_CONNECTIONS;
    ionModel.nodes = NULL;
    ionModel.numNodes = 0;
    ionModel.nextAvailableNodeId = 0;

    const char* dataSamples[] = {
        "The quick brown fox",
        "jumps over the lazy dog",
        "The quick blue hare",
        "leaps past the sleepy cat",
        "Bright stars twinkle above",
        "Soft winds stir the trees",
        "Rain falls on rooftops",
        "Children laugh in parks",
        "Old books gather dust",
        "New ideas spark minds"
    };
    int numSamples = sizeof(dataSamples) / sizeof(dataSamples[0]);

    simulateTextGrowth(&ionModel, dataSamples, numSamples);

    log_info("Cleaning up and exiting");
    for (int i = 0; i < ionModel.numNodes; i++) {
        free(ionModel.nodes[i].data);
        free(ionModel.nodes[i].lastData);
        free(ionModel.nodes[i].connections);
    }
    free(ionModel.nodes);
    if (__log_file) fclose(__log_file);

    return 0;
}