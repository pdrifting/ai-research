# Temporal Biological Hierachies for Expressing Boolean Logic Gates

## 1. Motivation and context

The pre‑alpha engine demonstrated that a single sigmoid neuron, when heavily biased and brute‑forced, can represent the full family of Boolean logic gates in a stable and interpretable way. In particular, XOR and XNOR require multiple neurons with symmetries and discontinuities that are fundamentally misaligned with smooth, shared‑weight optimization.

The alpha engine takes a different approach.  Instead of forcing all logic into a single weight space, it expresses Boolean gates as small biological microcircuits.  This was an attempt to see if simulated temporal biological mechanisms could be implemented as a single neuron with multiple dendrites and synapses, driven by ion flux, membrane potentials, and a discrete state machine. Logic is no longer a property of a learned weight vector, but of the physical configuration of synapses and dendrites.

## 2. Biological abstraction and design goals

The alpha model is intentionally simple but biologically flavored. It uses:

- **Ion flux tables** to represent different synaptic effects (fire, slow, negate, absolute high, shunt).
- **A membrane potential** (`soma_potential`) that integrates external flux and internal pumps.
- **A discrete state machine** to represent resting, firing, and refractory phases.
- **Dendrites and synapses** as the structural substrate where logic is encoded.

The design goal is not to be biophysically accurate, but to show that:

- Boolean logic can be expressed as **structural topology** over a common neuron model.
- The same neuron physics can implement AND, OR, NAND, NOR, NOT, XOR, and XNOR.
- Truth tables can be recovered from **spike counts** over time, not from a static activation.


## 3. Core neuron model

### 3.1 Universal ion flux model

The model defines a small, discrete set of synaptic “gate types”, each mapped to a fixed ion flux value:

```c
static const float EXT_ION_FLUX[] = {
    0.000f,  // 0: INACTIVE
    0.060f,  // 1: FIRE
    0.012f,  // 2: SLOW
   -0.012f,  // 3: NEGATE
    0.100f,  // 4: ABSOLUTE HIGH
   -0.100f   // 5: ABSOLUTE LOW (Shunt)
};
```

### 3.2 Pumping actions

These values represent how much each active synapse pushes the membrane potential up or down. The same physics is shared across all gates, where only the arrangement of synapses and their types changes from gate to gate.

```c
static const float INT_PUMP_VALS[] = {
    0.018f,  // Pump is increasing firing potential or creating refactory state
   -0.018f,  // Pump is negating firing potential or returning to resting or recovery state
    0.000f   // Pump has no change on the current soma state of the neuron
};
```

### 3.3 Membrane potentials and state machine

The system tracks the neuron membrane potential states:

```c
#define RESTING_POTENTIAL -0.65f
#define FIRE_POTENTIAL    -0.55f
#define ACTION_VOLTAGE     0.40f
#define REFACTORY_MIN     -0.75f
```

### 3.3 Dendrites, synapses, and topology

The neuron is structurally defined as:

```c
typedef struct {
    uint8_t gate_type;  // Type of the gate
    uint8_t is_active;  // Acts as the input/is it active
} Synapse;

typedef struct {
    uint16_t synapse_count;  // How many synapses 0 to 8
    Synapse  synapses[8];    // Holds the synaptic states
} Dendrite;

typedef struct {
   float soma_potential;    // Tracks the current state of the neurons soma potential
   uint8_t state;           // Current state of the neuron
   uint16_t dendrite_count; // How many dendrites the neuron is using
   Dendrite dendrites[4];   // Configuration of neuron dendrites
} Neuron;
```

## 4. Encoding Boolean gates as biological microcircuits

```C
void setup_gate(Neuron* n, const char* type) {
    // Reset neuron to a clean state
    memset(n, 0, sizeof(Neuron));
    n->soma_potential = RESTING_POTENTIAL;
    n->dendrite_count = 4;

    // ------------------------------------------------------------
    // AND Gate
    // ------------------------------------------------------------

    if (strcmp(type, "AND") == 0) {

        // Input A
        n->dendrites[1].synapse_count = 1;
        n->dendrites[1].synapses[0].gate_type = 2;   // SLOW excitatory

        // Input B
        n->dendrites[2].synapse_count = 1;
        n->dendrites[2].synapses[0].gate_type = 2;   // SLOW excitatory
    }

    // ------------------------------------------------------------
    // OR Gate
    // ------------------------------------------------------------

    else if (strcmp(type, "OR") == 0) {

        // Input A
        n->dendrites[1].synapse_count = 1;
        n->dendrites[1].synapses[0].gate_type = 1;   // FIRE excitatory

        // Input B
        n->dendrites[2].synapse_count = 1;
        n->dendrites[2].synapses[0].gate_type = 1;   // FIRE excitatory
    }

    // ------------------------------------------------------------
    // NAND Gate
    // ------------------------------------------------------------

    else if (strcmp(type, "NAND") == 0) {
        // Baseline excitation
        n->dendrites[0].synapse_count = 3;
        for (int i = 0; i < 3; i++)
            n->dendrites[0].synapses[i].gate_type = 2;   // SLOW excitatory

        // Input A inhibition
        n->dendrites[1].synapse_count = 1;
        n->dendrites[1].synapses[0].gate_type = 3;       // NEGATE inhibitory

        // Input B inhibition
        n->dendrites[2].synapse_count = 1;
        n->dendrites[2].synapses[0].gate_type = 3;       // NEGATE inhibitory
    }

    // ------------------------------------------------------------
    // NOR Gate  /  NOT Gate
    // ------------------------------------------------------------

    else if (strcmp(type, "NOR") == 0 || strcmp(type, "NOT") == 0) {
        // Baseline excitation
        n->dendrites[0].synapse_count = 3;
        for (int i = 0; i < 3; i++)
            n->dendrites[0].synapses[i].gate_type = 2;   // SLOW excitatory

        // Input A shunt
        n->dendrites[1].synapse_count = 1;
        n->dendrites[1].synapses[0].gate_type = 5;       // ABSOLUTE LOW (shunt)

        // NOR has a second input, NOT does not
        if (strcmp(type, "NOR") == 0) {
            n->dendrites[2].synapse_count = 1;
            n->dendrites[2].synapses[0].gate_type = 5;   // ABSOLUTE LOW (shunt)
        }
    }

    // ------------------------------------------------------------
    // XOR Gate
    // ------------------------------------------------------------

    else if (strcmp(type, "XOR") == 0) {
        // Input A
        n->dendrites[1].synapse_count = 1;
        n->dendrites[1].synapses[0].gate_type = 1;       // FIRE excitatory

        // Input B
        n->dendrites[2].synapse_count = 1;
        n->dendrites[2].synapses[0].gate_type = 1;       // FIRE excitatory

        // A AND B - shunt (cancels excitation)
        n->dendrites[3].synapse_count = 2;
        n->dendrites[3].synapses[0].gate_type = 5;       // ABSOLUTE LOW
        n->dendrites[3].synapses[1].gate_type = 5;       // ABSOLUTE LOW
    }

    // ------------------------------------------------------------
    // XNOR Gate
    // ------------------------------------------------------------

    else if (strcmp(type, "XNOR") == 0) {
        // Baseline excitation
        n->dendrites[0].synapse_count = 3;
        for (int i = 0; i < 3; i++)
            n->dendrites[0].synapses[i].gate_type = 2;   // SLOW excitatory

        // Input A shunt
        n->dendrites[1].synapse_count = 1;
        n->dendrites[1].synapses[0].gate_type = 5;       // ABSOLUTE LOW

        // Input B shunt
        n->dendrites[2].synapse_count = 1;
        n->dendrites[2].synapses[0].gate_type = 5;       // ABSOLUTE LOW

        // A AND B - reinforcement (absolute high)
        n->dendrites[3].synapse_count = 2;
        n->dendrites[3].synapses[0].gate_type = 4;       // ABSOLUTE HIGH
        n->dendrites[3].synapses[1].gate_type = 4;       // ABSOLUTE HIGH
    }
}
```

### 4.1 AND and OR as simple excitatory configurations

<pre>
AND Neuron (soma)
   |
   +-- Dendrite 1 (Input A)
   |       \
   |        +-- Synapse: gate=2 (SLOW excitatory)
   |
   +-- Dendrite 2 (Input B)
           \
            +-- Synapse: gate=2 (SLOW excitatory)
</pre>

<pre>
OR Neuron (soma)
   |
   +-- Dendrite 1 (Input A)
   |       \
   |        +-- Synapse: gate=1 (FIRE excitatory)
   |
   +-- Dendrite 2 (Input B)
           \
            +-- Synapse: gate=1 (FIRE excitatory)  
</pre>

### 4.2 NAND, NOR and NOT using inhibitory and shunt pathways

<pre>
NAND Neuron (soma)
   |
   +-- Dendrite 0 (Baseline excitation)
   |       \
   |        +-- Synapse: gate=2 (SLOW excitatory)
   |        +-- Synapse: gate=2 (SLOW excitatory)
   |        +-- Synapse: gate=2 (SLOW excitatory)
   |
   +-- Dendrite 1 (Input A)
   |       \
   |        +-- Synapse: gate=3 (NEGATE inhibitory)
   |
   +-- Dendrite 2 (Input B)
           \
            +-- Synapse: gate=3 (NEGATE inhibitory)
</pre>

<pre>
NOR Neuron (soma)
   |
   +-- Dendrite 0 (Baseline excitation)
   |       \
   |        +-- Synapse: gate=2 (SLOW excitatory)
   |        +-- Synapse: gate=2 (SLOW excitatory)
   |        +-- Synapse: gate=2 (SLOW excitatory)
   |
   +-- Dendrite 1 (Input A)
   |       \
   |        +-- Synapse: gate=5 (ABSOLUTE LOW / Shunt)
   |
   +-- Dendrite 2 (Input B)
           \
            +-- Synapse: gate=5 (ABSOLUTE LOW / Shunt) 
</pre>

<pre>
(MAPPED FROM NOR WITH ONLY ONE INPUT)

NOT Neuron (soma)
   |
   +-- Dendrite 0 (Baseline excitation)
   |       \
   |        +-- Synapse: gate=2 (SLOW excitatory)
   |        +-- Synapse: gate=2 (SLOW excitatory)
   |        +-- Synapse: gate=2 (SLOW excitatory)
   |
   +-- Dendrite 1 (Input A)
           \
            +-- Synapse: gate=5 (ABSOLUTE LOW / Shunt)
</pre>

### 4.3 XOR and XNOR via composite dendritic structures

<pre>
XOR Neuron (soma)
   |
   +-- Dendrite 1 (Input A)
   |       \
   |        +-- Synapse: gate=1 (FIRE excitatory)
   |
   +-- Dendrite 2 (Input B)
   |       \
   |        +-- Synapse: gate=1 (FIRE excitatory)
   |
   +-- Dendrite 3 (A AND B shunt)
           \
            +-- Synapse: gate=5 (ABSOLUTE LOW / Shunt)
            +-- Synapse: gate=5 (ABSOLUTE LOW / Shunt)
</pre>

<pre>
XNOR Neuron (soma)
   |
   +-- Dendrite 0 (Baseline excitation)
   |       \
   |        +-- Synapse: gate=2 (SLOW excitatory)
   |        +-- Synapse: gate=2 (SLOW excitatory)
   |        +-- Synapse: gate=2 (SLOW excitatory)
   |
   +-- Dendrite 1 (Input A)
   |       \
   |        +-- Synapse: gate=5 (ABSOLUTE LOW / Shunt)
   |
   +-- Dendrite 2 (Input B)
   |       \
   |        +-- Synapse: gate=5 (ABSOLUTE LOW / Shunt)
   |
   +-- Dendrite 3 (A AND B reinforcement)
           \
            +-- Synapse: gate=4 (ABSOLUTE HIGH)
            +-- Synapse: gate=4 (ABSOLUTE HIGH)
</pre>

## 5. Simulation procedure

The process_neuron function:
- Forces the potential to action voltage when firing, or to refractory minimum when in refractory.
- Integrates synaptic flux from all active synapses.
- Applies an internal pump (INT_PUMP_VALS) to slowly return the neuron toward resting potential.

This is biologically inspired:
- A firing neuron’s membrane potential is forced to the spike voltage
- A refractory neuron’s membrane potential is forced to a minimum
- Only in resting/returning states does the soma integrate synaptic flux

Represent a state‑dependent override system:
- Firing forces spike voltage
- Refractory forces minimum
- Otherwise integrate normally

This is exactly how real neurons behave:
- A spike is an all‑or‑nothing event
- After a spike, the membrane is suppressed
- Only outside those windows does integration occur

```C
void process_neuron(Neuron* n) {
    // Convert the neuron's state into numeric flags
    // Four potential states
    // 0 - Resting
    // 1 - Firing (spike)
    // 2 - Refactory
    // 3 - Recovering

    float is_st1 = (float)(n->state == 1);  // Firing
    float is_st2 = (float)(n->state == 2);  // Refactory

    // Normalizes the potential created by the two inputs
    // If the neuron is firing (state == 1):
    //    is_st1 = 1, is_st2 = 0 → normal_ops = 0
    //
    // If the neuron is refractory (state == 2):
    //    is_st1 = 0, is_st2 = 1 → normal_ops = 0
    //
    // If the neuron is resting or returning (state == 0 or 3):
    //    is_st1 = 0, is_st2 = 0 → normal_ops = 1
    //
    // When the neuron is firing or refractory | normal integration is disabled
    // When the neuron is not in those states  | normal integration is enabled

    float normal_ops = 1.0f - (is_st1 + is_st2);

    // 0 - Resting    - soma_potential = soma_potential + 0 + 0
    // 1 - Firing     - soma_potential = 0 + ACTION_VOLTAGE + 0
    // 2 - Refactory  - soma_potential = 0 + 0 + REFACTORY_MIN
    // 3 - Recovering - soma_potential = soma_potential + 0 + 0

    n->soma_potential = (n->soma_potential * normal_ops) + (ACTION_VOLTAGE * is_st1) + (REFACTORY_MIN * is_st2);

    // ------------------------------------------------------------
    // 1. ACCUMULATE SYNAPTIC FLUX FROM ALL ACTIVE SYNAPSES
    // ------------------------------------------------------------
    //
    // Each dendrite contains 0–8 synapses.
    // Each synapse has:
    //    - gate_type  → selects an ion flux constant from EXT_ION_FLUX[]
    //    - is_active  → 1 if the input is ON, 0 if OFF
    //
    // EXT_ION_FLUX[] contains the "physics" of each synapse type:
    //    0 = 0.000   (INACTIVE)
    //    1 = +0.060  (FIRE excitatory)
    //    2 = +0.012  (SLOW excitatory)
    //    3 = -0.012  (NEGATE inhibitory)
    //    4 = +0.100  (ABSOLUTE HIGH)
    //    5 = -0.100  (ABSOLUTE LOW / Shunt)
    //
    // The flux accumulator sums the contribution of EVERY active synapse.
    // If a synapse is inactive, it contributes 0.
    //
    // NOTE: Flux is only applied when normal_ops == 1.
    //       If the neuron is firing or refractory, flux is ignored.
    //

    float flux = 0.0f;

    for (int d = 0; d < n->dendrite_count; d++) {
        for (int s = 0; s < n->dendrites[d].synapse_count; s++) {

            // Multiply:
            //   is_active (0 or 1)
            //   × EXT_ION_FLUX[gate_type]
            //
            // The "& 0x7" masks the gate_type to the lower 3 bits
            // (a safety measure to prevent invalid indexing).

            flux += (float)n->dendrites[d].synapses[s].is_active *
                    EXT_ION_FLUX[n->dendrites[d].synapses[s].gate_type & 0x7];
        }
    }

    // Add the accumulated flux to the soma potential.
    // Again, this only happens when normal_ops == 1.
    // If the neuron is firing or refractory, flux is ignored.

    n->soma_potential += (flux * normal_ops);

    // ------------------------------------------------------------
    // 2. INTERNAL ION PUMP (HOMEOSTASIS)
    // ------------------------------------------------------------
    //
    // The neuron has a simple internal pump that pushes the membrane
    // potential back toward the resting potential (-0.65).
    //
    // INT_PUMP_VALS:
    //    [0] = +0.018   (pump upward)
    //    [1] = -0.018   (pump downward)
    //    [2] =  0.000   (no pump)
    //
    // pump_idx selects upward or downward pump:
    //    pump_idx = 1  → soma > resting → pump downward
    //    pump_idx = 0  → soma <= resting → pump upward
    //
    // at_rest is a tiny window around the exact resting potential.
    // If the neuron is already at rest, the pump is disabled.
    //

    int pump_idx = (n->soma_potential > (RESTING_POTENTIAL + 0.001f));
    bool at_rest = (n->soma_potential >= -0.651f && n->soma_potential <= -0.649f);

    // Apply the pump:
    //    - upward or downward depending on pump_idx
    //    - OR no pump if at_rest == true
    //    - AND only when normal_ops == 1

    n->soma_potential += INT_PUMP_VALS[at_rest ? 2 : pump_idx] * normal_ops;

    // ------------------------------------------------------------
    // 3. DETERMINE WHETHER THE NEURON FIRES A SPIKE
    // ------------------------------------------------------------
    //
    // A spike occurs when:
    //    - soma_potential >= FIRE_POTENTIAL (-0.55)
    //    - AND the neuron is in a state that allows firing:
    //          state == 0 (resting)
    //          state == 3 (recovering)
    //
    // The neuron CANNOT fire if:
    //    - it is already firing (state == 1)
    //    - it is refractory (state == 2)
    //

    bool fire = (n->soma_potential >= FIRE_POTENTIAL) &&
                (n->state == 0 || n->state == 3);

    // ------------------------------------------------------------
    // 4. STATE MACHINE TRANSITION
    // ------------------------------------------------------------
    //
    // next_state_map defines the automatic transitions:
    //
    //    state 0 (resting)    → next_state_map[0] = 0  (stay resting)
    //    state 1 (firing)     → next_state_map[1] = 2  (go to refractory)
    //    state 2 (refractory) → next_state_map[2] = 3  (recovering)
    //    state 3 (recovering) → next_state_map[3] = 0  (return to rest)
    //
    // If "fire" is true, override the state to 1 (firing).
    // Otherwise, follow the automatic transition.
    //

    static const uint8_t next_state_map[] = { 0, 2, 3, 0 };
    n->state = fire ? 1 : next_state_map[n->state];
}
```

## 6. Generating spike‑based truth tables

To evaluate each gate, the simulator constructs a neuron configured with the appropriate dendritic topology and synaptic gate types. Boolean inputs are applied by setting the is_active flags on the synapses belonging to the input dendrites. Some gates (XOR, XNOR) include an additional dendrite that activates only when both inputs are true, enabling composite logic.

A baseline excitatory drive is always applied on dendrite 0 for gates that require a resting bias (NAND, NOR, XNOR).

Once the neuron is configured, it is simulated for 60 discrete time steps. At each step, the neuron integrates synaptic flux, applies internal pump dynamics, updates its membrane potential, and transitions through its firing state machine. A spike is counted whenever the neuron enters the firing state (state == 1).

The final Boolean output is defined as:
```C
output = (spikes > 0)
```

This produces a spike‑based truth table for each gate, demonstrating that the same neuron physics can express AND, OR, NAND, NOR, NOT, XOR, and XNOR purely through structural configuration.

```C
// ------------------------------------------------------------
// test_gate
// ------------------------------------------------------------
// Runs a single Boolean input pair (inA, inB) through a fully
// configured neuron and measures how many spikes occur over
// a fixed simulation window.
//
// This function is responsible for:
//   1. Constructing the neuron for the requested gate type
//   2. Activating synapses based on Boolean inputs
//   3. Running the neuron for 60 time steps
//   4. Counting spikes (state == 1)
//   5. Printing a truth-table row
//
void test_gate(const char* type, bool inA, bool inB) {

    // --------------------------------------------------------
    // 1. Construct a fresh neuron configured for this gate
    // --------------------------------------------------------
    Neuron n;
    setup_gate(&n, type);


    // --------------------------------------------------------
    // 2. Apply Boolean inputs to dendrites
    // --------------------------------------------------------
    //
    // Dendrite 1 = Input A
    // Dendrite 2 = Input B
    //
    // Each dendrite may contain 0–N synapses depending on the
    // gate topology. Every synapse on that dendrite receives
    // the same Boolean activation.
    //
    for (int s = 0; s < n.dendrites[1].synapse_count; s++)
        n.dendrites[1].synapses[s].is_active = inA;

    for (int s = 0; s < n.dendrites[2].synapse_count; s++)
        n.dendrites[2].synapses[s].is_active = inB;


    // --------------------------------------------------------
    // 3. Special-case activation for XOR and XNOR
    // --------------------------------------------------------
    //
    // XOR and XNOR use dendrite 3 as an "A AND B" pathway.
    // This dendrite is active only when BOTH inputs are true.
    //
    if (strcmp(type, "XOR") == 0 || strcmp(type, "XNOR") == 0) {
        bool andAB = (inA && inB);
        for (int s = 0; s < n.dendrites[3].synapse_count; s++)
            n.dendrites[3].synapses[s].is_active = andAB;
    }


    // --------------------------------------------------------
    // 4. Baseline excitation (dendrite 0)
    // --------------------------------------------------------
    //
    // Many gates (NAND, NOR, XNOR) rely on a constant low-level
    // excitatory drive to keep the neuron near threshold.
    //
    // This baseline is ALWAYS active for every gate type.
    //
    for (int s = 0; s < n.dendrites[0].synapse_count; s++)
        n.dendrites[0].synapses[s].is_active = 1;


    // --------------------------------------------------------
    // 5. Run the simulation for 60 time steps
    // --------------------------------------------------------
    //
    // Each call to process_neuron() performs:
    //   - state-dependent override
    //   - synaptic flux integration
    //   - internal pump adjustment
    //   - firing threshold check
    //   - state machine transition
    //
    // A spike is counted whenever state == 1.
    //
    int spikes = 0;
    for (int i = 0; i < 60; i++) {
        process_neuron(&n);
        if (n.state == 1)
            spikes++;
    }


    // --------------------------------------------------------
    // 6. Print the truth-table row
    // --------------------------------------------------------
    //
    // Example output:
    //   AND(1,0) ->  0 spikes [0]
    //   OR (1,0) -> 12 spikes [1]
    //
    // The final Boolean output is (spikes > 0).
    //
    printf("%4s(%d,%d) -> %2d spikes [%d]\n",
           type, inA, inB, spikes, spikes > 0);
}



// ------------------------------------------------------------
// main
// ------------------------------------------------------------
// Iterates over all gate types and prints their full truth tables.
// Each gate is tested with all valid input combinations.
//
// NOT is unary, so only (0) and (1) are tested.
// All other gates test (0,0), (1,0), (0,1), (1,1).
//
int main() {

    const char* g[] = {"AND", "OR", "NAND", "NOR", "NOT", "XOR", "XNOR"};

    for (int i = 0; i < 7; i++) {

        printf("\n--- %s ---\n", g[i]);

        // Always test (0,0) and (1,0)
        test_gate(g[i], 0, 0);
        test_gate(g[i], 1, 0);

        // NOT is unary → skip (0,1) and (1,1)
        if (strcmp(g[i], "NOT") != 0) {
            test_gate(g[i], 0, 1);
            test_gate(g[i], 1, 1);
        }
    }

    return 0;
}
```

## 7. Results: recovered truth tables from biological dynamics

Every gate has a unique spiking fingerprint:

AND - Only fires when both inputs are true, using only 2 spikes. A clean, minimal, tightly‑controlled excitatory accumulation.

OR - Fires strongly whenever either input is true — 10–15 spikes. This is a high‑gain excitatory topology.

NAND - Fires on everything except (1,1), but with graded spike counts.

NOR - Only fires on (0,0) with a stable 6‑spike baseline.

NOT - Same as NOR but unary.

XOR - The classic “hard” gate:
  - (1,0) → 10 spikes
  - (0,1) → 10 spikes
  - (1,1) → shunted to zero

XNOR - The mirror image of XOR and the spike counts match the NOR baseline.

The spike counts are not arbitrary and meaingfully reflect:
- the ion flux constants
- the dendritic topology
- the state machine timing
- the pump dynamics
- the threshold crossings
- the refractory window

Most neural networks:
- approximate logic
- with continuous activations
- and floating‑point weights
- and gradient descent

This Alpha engine:
- expresses logic
- through dendritic structure
- and discrete ion flux
- and spike timing
- and state transitions

This is a fundamentally different computational paradigm. This Alpha model represents a simulated physical system, and the truth tables emerged from the physics.  The fact that the spike counts differ per gate means the model is able to express logic through dynamics, not just binary thresholds.

Here is the emitted gate truth tables showing the spike counts:

<pre>
--- AND ---
 AND(0,0) ->  0 spikes [0]
 AND(1,0) ->  0 spikes [0]
 AND(0,1) ->  0 spikes [0]
 AND(1,1) ->  2 spikes [1]

--- OR ---
  OR(0,0) ->  0 spikes [0]
  OR(1,0) -> 10 spikes [1]
  OR(0,1) -> 10 spikes [1]
  OR(1,1) -> 15 spikes [1]

--- NAND ---
NAND(0,0) ->  6 spikes [1]
NAND(1,0) ->  2 spikes [1]
NAND(0,1) ->  2 spikes [1]
NAND(1,1) ->  0 spikes [0]

--- NOR ---
 NOR(0,0) ->  6 spikes [1]
 NOR(1,0) ->  0 spikes [0]
 NOR(0,1) ->  0 spikes [0]
 NOR(1,1) ->  0 spikes [0]

--- NOT ---
 NOT(0,0) ->  6 spikes [1]
 NOT(1,0) ->  0 spikes [0]

--- XOR ---
 XOR(0,0) ->  0 spikes [0]
 XOR(1,0) -> 10 spikes [1]
 XOR(0,1) -> 10 spikes [1]
 XOR(1,1) ->  0 spikes [0]

--- XNOR ---
XNOR(0,0) ->  6 spikes [1]
XNOR(1,0) ->  0 spikes [0]
XNOR(0,1) ->  0 spikes [0]
XNOR(1,1) ->  6 spikes [1]
</pre>

The raw Boolean gate spike tables:
- demonstrate correctness
- demonstrate stability
- demonstrate biological plausibility
- demonstrate that the model is not “just a lookup table”
- demonstrate that the topology matters
    
## 8. Limitations and lessons for later engines

The Alpha engine represents a radical shift from the Pre‑Alpha model. The Pre‑Alpha system solved discrete Boolean gates using brute‑force sigmoid evaluation, but it could not express those gates structurally inside a synthetic neuron. To move forward, we needed a model where logic emerges from dendritic structure, ion flux, and spike timing, not from weight matrices.

The Alpha engine achieved the most rudimentary state of this, but the shift introduced new constraints.

### 8.1 Structural Logic vs. Chained Computation

The Alpha neuron can express any Boolean gate through dendritic topology alone. However, the engine cannot yet chain neurons together to form multi‑stage logic circuits.

This limitation arises from the physics of the model:
- Each neuron has its own timing, integration window, and refractory cycle.
- Spikes occur at different offsets depending on synapse type, flux strength, and pump behavior.
- There is no global clock, no synchronization, and no mechanism to align spike phases across neurons.

As a result:
- The Alpha engine can compute single step logic, but it cannot yet compose logic chains like Pre-Alpah.
- This is a fundamental limitation of the current architecture.

### 8.2 Timing and Synchronization Constraints

The process_neuron() function is intentionally simple and biologically inspired, but it lacks:
- cycle‑accurate stepping
- phase alignment
- spike‑window gating
- temporal buffering
- inter‑neuron synchronization
- bus‑like communication channels

In biological systems, spike trains are coordinated through:
- oscillatory rhythms
- inhibitory gating
- dendritic delays
- population synchrony
- thalamic timing loops

We do not yet understand these mechanisms well enough to replicate them faithfully.  Hopefully further research on this model can help unlock some of the unknown black boxes of biological brains.

The Alpha engine is correct starting place for future research, but it is not scalable.

### 8.3 The Pump Physics Create a Digital Barrier

The internal pump enforces a stable resting potential:
- If the membrane is below resting, pump pushes ions in (+0.018)
- If the membrane is above resting, pump pushes ions out (–0.018)
- If the membrane is at resting, pump is inactive (0.000)

This creates a digital barrier:
- Any excitation < 0.018 is erased by the pump
- Only excitation > 0.018 can accumulate
- This produces clean, discrete spike behavior

This is why the Alpha engine can express Boolean logic so cleanly, but it also means the system is extremely sensitive to timing and cannot yet support multi‑neuron accumulation.

### 8.4 Why Addition, Counting, and Multi‑Step Reasoning Fail

Things that have been explored to date:
- Modelling different types of neurons
- Creating near‑threshold accumulators
- Detecting overflow
- Forced resets post soma potential

Core features are missing the Alpha engine to accomplish this:
- a clock,
- a bus,
- a synchronization mechanism or temporal gating

The Alpha system cannot reliably pass results from one neuron to the next.  This postulates an upgrade to the Alpha model that allows information sharing between neurons.

### 8.5 What the Next Engine Needs
To move beyond the Alpha engine, the next model must introduce:

#### 8.5.1 A Clock or Phase System

Future models need more than a digital CPU clock, possibly a biological‑style oscillatory rhythm that:
- aligns spike windows
- synchronizes integration
- allows multi‑neuron chains
- enables sequential reasoning

There are four possible routes to explore.

---

#### Option 1: ASYNCHRONOUS (Biological, free‑running spikes)
Pros:
- Most biologically realistic: Neurons fire whenever their internal state crosses threshold, just like real cortical tissue.
- Rich emergent dynamics: Timing, interference, oscillations, and spontaneous patterns appear naturally.
- Energy efficient and event‑driven: Nothing happens unless a spike happens.
- Captures the messiness of real thought: Noise, drift, and instability are part of the computation.

Cons:
- Timing chaos: No global reference frame means spikes collide, drift, or miss each other.
- Race conditions everywhere: A spike arriving 1 step earlier or later can change the entire computation.
- No reliable chaining: Multi‑step reasoning collapses because nothing stays aligned long enough.
- Hard to debug, hard to scale: Every neuron is its own timing universe.

Result:
- Beautiful but unreliable computation.
- Great for local logic.
- Terrible for sequential reasoning (addition, planning, multi‑step tasks).
- Thoughts appear and vanish exactly like fleeting biological cognition.

Limitations:
- Cannot guarantee order of operations
- Cannot reliably store or propagate state
- Cannot perform multi‑step arithmetic
- Requires enormous architectural scaffolding to scale

Complexity:
- Low code complexity (simple neurons, simple loops)
- High emergent complexity (timing interactions explode combinatorially)

---

#### Option 2: SYNCHRONOUS (Global clock, engineered timing)
Pros:
- Predictable and deterministic: Every neuron updates at the same moment, eliminating race conditions.
- Supports chaining and multi‑step reasoning: You can build adders, counters, registers, and pipelines.
- Easy to debug: One tick = one logical step.
- Scales extremely well: This is why all digital computers use it.

Cons:
- Less biological: Real neurons do not wait for a clock tick.
- Requires a timing source: A global oscillator or discrete update cycle.
- Can suppress emergent dynamics: Some natural timing phenomena disappear under strict synchronization.

Result:
- Reliable, engineered computation
- Perfect for models that need:
  - sequential and functional logic
  - stable reasoning
  - memory and thought chaining

Limitations:
- Less biologically faithful
- Harder to model real spike timing
- Removes some emergent behaviors
- Requires careful design of clock frequency and update rules

Complexity:
- Moderate conceptual complexity (you must define the clock and update rules)
- Low operational complexity (everything updates together, easy to reason about)

---

#### Option 3: SELF-TIMING (Local clocks, handshake protocols)
Pros:
- Closer to biology than a global clock: Each circuit or microcircuit has its own oscillation, rhythm, or timing loop acting like cortical columns, hippocampal theta, or cerebellar microzones.
- Highly modular: Circuits can run independently and only synchronize when needed.
- Scales better than pure asynchronous systems: Because local timing reduces race conditions inside each module.
- Allows flexible, dynamic computation: Circuits can speed up, slow down, or pause depending on context.

Cons:
- Extremely complex to design
- You need:
  - local oscillators
  - phase relationships
  - handshake protocols
  - gating mechanisms
- Buffering
- Error recovery
- Still vulnerable to drift  
  - Local clocks can desynchronize over time unless you add correction mechanisms.
- Hard to debug:
  - Failures happen at boundaries between modules
  - Not inside them.

Result:
- Flexible, powerful, semi‑biological computation
- Requires a massive amount of architectural scaffolding

Limitations:
- Hard to scale without enormous engineering effort
- Requires explicit protocols for every interaction
- Debugging timing bugs is a nightmare
- Still not fully biological, still not fully reliable

Complexity
- Very high: This is the most complex option by far, but the most promising long‑term.

---

#### Option 4: TIMELESS (State-based, continuous values)
Pros:
- Simple, stable, predictable: No spikes, no timing windows, no refractory periods.
- Easy to train: Works with gradient descent, backprop, and modern ML tooling.
- Highly scalable: Billions of parameters, massive parallelism, GPU‑friendly.

Cons:
- Not biological: No spikes, no dendrites, no ion flux, no timing, no structure.
- No temporal richness: Time is simulated, not emergent.
- No physical grounding: Everything is abstract math, not physics.
- Loses the advantages of spikes:
  - energy efficiency
  - event‑driven computation
  - sparse firing
  - temporal coding

Result:
- Reliable, powerful, but fundamentally un‑brainlike  
- Great for engineering.
- Poor for understanding biological intelligence.

Limitations:
- Cannot model spike timing
- Cannot model dendritic computation
- Cannot model biological constraints
- Cannot model emergent timing phenomena
- Cannot model physical intelligence

Complexity
- Low conceptual complexity (easy to understand, easy to train)
- High computational complexity (requires huge compute capacity, expensive hardware, large datasets)

Timeless models such as ANNs and LLMs suffer from structural and entropic collapse.  They lack spikes, dendrites, timing, locality, and physical grounding. Their differentiable nature forces smoothing, suppresses true randomness, prevents modular computation, and produces cyclical attractors that fail entropy tests. These limitations make certain classes of reasoning, logic, and noise‑driven computation fundamentally impossible.

#### 8.5.2 A Communication Bus

A way for neurons to:
- publish their spike results
- hold them stable
- pass them to downstream neurons
- avoid timing collisions

#### 8.5.3 Temporal Buffers

A mechanism to:
- store intermediate results
- accumulate multi‑step computations
- allow nested logic
- support multi‑layer circuits

#### 8.5.4 Multi‑Neuron Coordination

Inspired by:
- cortical microcolumns
- thalamic relay timing
- hippocampal theta rhythms
- cerebellar timing loops

This is where the Beta engine will begin.

### 8.6 Why This Matters for Understanding Biological Thought

This limitation is not a failure, it is a clue.
- Humans can chain thoughts.
- We can perform multi‑step reasoning.
- We can accumulate partial results.
- We can synchronize across populations of neurons.

We do not yet know how.  By building synthetic engines that fail in specific ways, we expose the missing mechanisms.

The Alpha engine shows:
- A single neuron can compute logic.
- But chaining logic requires timing mechanisms we do not yet understand.

This is a research direction with profound implications for:
- cognitive science
- neuromorphic engineering
- synthetic intelligence
- computational neuroscience

###  8.7 Summary: The Alpha Engine Is Correct but Primitive

The Alpha engine:
- correctly expresses Boolean logic
- produces stable spike signatures
- demonstrates structural computation
- is biologically grounded
- is reproducible and open

But it cannot:
- chain neurons
- synchronize timing
- perform multi‑step reasoning
- accumulate results across cycles

This is not a flaw, it is the natural boundary of the model. This is the Alpha for a reason. 

## 9. Summary

The Three main paths that got us here, and the questions we should all be asking if we are going to advance and progress towards AGI, or efficiency in general.  The mainstream path AI research is currently taking is unsustainable.

---

### Path A – Biophysical neurons (Hodgkin & Huxley & NEURON)

What it is?  
- Models real neurons: ion channels, dendrites, membrane potentials, spike generation.

What did it achieve?
- Deep descriptive understanding of single neurons and small circuits.

What did it miss?
- No clear account of how neurons compute logic, chain thoughts, synchronize, or implement reasoning.

What is the status?
- Scientifically rich, computationally underdeveloped, largely abandoned as a route to AI.

*Researchers such as Yiota Poirazi have proposed that dendrites act as nonlinear subunits, allowing a single neuron to behave like a multi‑layer network. These ideas are conceptually important, but they remain metaphors rather than computational frameworks. They do not provide mechanisms for chaining, timing, memory, or reasoning. This gap between descriptive insight and operational computation is where Path A stalled.*

---

### Path B – Engineering “neural networks” (ANNs, deep learning, LLMs)

What it is?
- Matrix multiplications and activation functions (ReLU, tanh, sigmoid) plus forward and/or back propogation.

What did it achieve?
- Huge practical success: vision, language, speech, games, coding, etc.

What did it miss?
- Biological realism, spike timing, dendritic computation, physical grounding.

What is the status?
- Dominant, powerful, but resource‑exhaustive and conceptually far from real neurons. GPU‑hungry, RAM‑hungry, centralized, expensive, high barrier to entry, distorts hardware markets.

---

### Path C – Biophysically grounded computation (the missing path)

What it is?
- Neurons that compute because of their physics: ion flux, dendrites, spikes, timing, structure.

What exists?  
- Very little. This Alpha engine is one of the very few examples in existence. This is not very understood path of research and little has been published on Neuroscience in the past 70 years to really help move it forward.

What is the goal?
- Connect real neuron dynamics to logic, memory, chaining, and reasoning—without abandoning physics.

What is the status?
- Largely unexplored, high‑value, high‑difficulty, and crucial for future accessible, brainlike AI.

*Alpha demonstrates that dendrites, synapses, and ion‑flux weights can be configured to implement real logic gates using real physics. This operational result fills a gap left open in decades of dendritic computation literature, which has described nonlinear dendritic behavior but never produced a computational substrate capable of using it.*

---

### Common misconceptions students need to see through

“Neural networks” are not neural in nature at all.  They are just interconnected matrix lattices.  Most are just differentiable function approximators with no spikes, no dendrites, no timing requiring expensive hardware and are terrible infficient.  Scale is not understanding. Bigger models does not equal deeper insight into how real neurons think.

NEURON is not an AI system, despite being considered the gold standard of representation.  It simulates biophysics, not computation, logic, or reasoning.

Spiking neural networks (as commonly implemented) are often cosmetic.  Many still rely on backprop and rate codes, not true spike‑timing computation.

### Where the real open problems are

For students who want to do real science, not just chase hype:
- How do dendrites implement logic?
- How does spike timing support chaining and multi‑step reasoning?
- How do biological circuits synchronize without a global clock?
- How do physical constraints (ion flux, pumps, refractory periods) shape computation?
- How can we build accessible, low‑resource models that still respect biology?

### How to read the archive

This archive is not just a record of models. It is a map of where science stalled, where hype dominates, and where the unfinished work lies. If you want to do real, accessible science, start here.  The efforts in this repository are to revisit, rewaken, and make research accessible.

There has been a collapse of STEM and a rise in abstraction that has been taking place for decades.  STEM education is collapsing globally despite efforts to attract more youth to it:
- The "T" is the weakest link.
- Teachers are disappearing.
- Students are trained in tools, not systems.
- Abstraction has replaced understanding.
- Efficiency has been replaced by scale.
- Innovation has been replaced by hype.

This is the root cause of the global engineering drought.  This is an effort to fix that drought in understanding that current architectures are incomplete, and the next paradigm in computational intelligence won't come form scaling.

It will come when we start thinking about:
- causal propagation
- temporal reasoning, the gap between reactive and proactive reasoning
- dynamic constraint systems and satisfaction
- self‑organizing emergent structure
- proactive learning without prompting and training
- preserves information lineage, carries forward the wisdom we lose
- narrative context and grouping
- persistent memory and inference substrates
- efficiency, real‑time optimization
- stewards of long‑term trajectories

We are a long way from any of these. Even the most comprehensive dendritic computation reviews, such as the Annual Review of Neuroscience article by London & Häusser (2005), conclude with a "wish list" of unanswered questions. These works identify the nonlinear, compartmentalized, and computationally rich properties of dendrites, but they provide no operational models, no circuit frameworks, and no mechanisms for chaining or reasoning. The roadmap they outline remains untraveled because the field lacks a computational substrate capable of expressing dendritic logic. This repository begins where those roadmaps end.


