import random
import numpy as np
from collections import deque

class ClockModule:
    def __init__(self, interval=1):
        """
        interval: Number of cycles between pulses. 
        1 = Every cycle (1111)
        2 = Every other cycle (1010)
        4 = Every 4th cycle (10001000)
        """
        self.interval = interval
        self.cycle_count = 0
        self.is_high = False

    def tick(self):
        """Advances the clock by one virtual cycle."""
        self.cycle_count += 1
        # Regulation logic: Pulse HIGH only on the interval boundary
        if self.cycle_count % self.interval == 0:
            self.is_high = True
        else:
            self.is_high = False
        return 1 if self.is_high else 0

    def set_frequency(self, new_interval):
        """Allows external regulation of the clock speed."""
        self.interval = max(1, new_interval)

class ProbabilisticGate:
    def __init__(self, pass_likelihood=0.5):
        """
        pass_likelihood: Float between 0.0 and 1.0 
        (e.g., 0.8 means 80% chance to pass the input)
        """
        self.threshold = pass_likelihood

    def process(self, input_bit):
        """
        Determines if the input_bit is allowed to pass through
        or if it is blocked (outputting a 0).
        """
        # Roll for the likelihood
        if random.random() < self.threshold:
            return input_bit  # Pass
        else:
            return 0          # Block
            
    def set_threshold(self, new_likelihood):
        """Updates the internal pass fraction."""
        self.threshold = max(0.0, min(1.0, new_likelihood))

class PulseJunction:
    def __init__(self, num_outputs=2):
        """
        num_outputs: The number of identical signals to emit.
        """
        self.num_outputs = num_outputs

    def distribute(self, input_bit):
        """
        Takes one input bit and returns a list of identical bits.
        """
        # Logic: Exactly replicate the input to all output lines
        return [input_bit for _ in range(self.num_outputs)]
    
    def set_fan_out(self, n):
        """Adjusts the number of output channels."""
        self.num_outputs = max(1, n)

class AccumulatingEnvelope:
    def __init__(self, pattern=[1, 1, 1], n_outputs=1):
        """
        pattern: The target bit-sequence to match.
        n_outputs: Number of distribution ports.
        """
        self.pattern = list(pattern)
        # Initialize window with zeros
        self.window = [0] * len(pattern)
        self.n_outputs = n_outputs

    def receive(self, input_bit):
        """
        Shifts the window to the right and inserts the new bit at index 0.
        """
        # 1. Right Shift: Move everything over
        # [0, 1, 1] becomes [?, 0, 1]
        for i in range(len(self.window) - 1, 0, -1):
            self.window[i] = self.window[i-1]
            
        # 2. Insert at Head
        # [?, 0, 1] becomes [1, 0, 1]
        self.window[0] = input_bit
        
        # 3. Pattern Match Check
        if self.window == self.pattern:
            return [1] * self.n_outputs
        else:
            return [0] * self.n_outputs

class OscillatoryCore:
    def __init__(self, n=48):
        """
        n: The number of nodes (state width).
        """
        self.n = n
        # Internal state vector
        self.state = np.random.randint(0, 2, n)
        # Topology: Which nodes affect which?
        # A simple rotation + feedback ensures oscillation.
        self.indices = np.arange(n)

    def tick(self, input_vector):
        """
        Processes n inputs to update n internal states and emit n outputs.
        input_vector: A list or array of length n.
        """
        # 1. Right-Shift the internal state (The Rotation)
        new_state = np.roll(self.state, 1)
        
        # 2. Perturbation Logic:
        # Every node is updated by XORing its current state with:
        # - The state of the node before it (the roll)
        # - The external input bit for that specific node
        for i in range(self.n):
            # Non-linear oscillation logic
            # output = (Neighbor) XOR (External Input)
            new_state[i] = new_state[i] ^ input_vector[i]
            
        # 3. Global Inversion: 
        # To ensure it never dies (0000), we flip the head if the tail is 0.
        if new_state[-1] == 0:
            new_state[0] = new_state[0] ^ 1

        self.state = new_state
        return self.state.tolist()

    def get_state(self):
        return self.state.tolist()

class ConfigurableOscillator:
    def __init__(self, n=48):
        self.n = n
        self.state = np.random.randint(0, 2, n)
        
        # Each input port has a 'target_width' and a 'current_count'
        # Default: All ports flip on every 1st pulse (width=1)
        self.pulse_widths = np.ones(n, dtype=int)
        self.pulse_counters = np.zeros(n, dtype=int)

    def set_port_width(self, port_index, width):
        """Configures a specific port to only flip every 'width' pulses."""
        if 0 <= port_index < self.n:
            self.pulse_widths[port_index] = max(1, width)
            self.pulse_counters[port_index] = 0

    def tick(self, input_vector):
        """
        Processes inputs based on pulse-width logic and rotates state.
        """
        # 1. Logic Processing for each Input
        for i in range(self.n):
            if input_vector[i] == 1:
                self.pulse_counters[i] += 1
                
                # Check if we hit the pulse-width threshold
                if self.pulse_counters[i] >= self.pulse_widths[i]:
                    # FLIP the internal bit
                    self.state[i] = 1 if self.state[i] == 0 else 0
                    # Reset counter
                    self.pulse_counters[i] = 0
        
        # 2. Internal Oscillation (Rotation)
        # This ensures nodes influence each other even without input
        self.state = np.roll(self.state, 1)
        
        # 3. Prevent Deadlock (Always keep the ring moving)
        if np.all(self.state == 0):
            self.state[0] = 1
            
        return self.state.tolist()

class QueuedJoiner:
    def __init__(self, capacity=1024):
        self.queue = deque(maxlen=capacity)

    def ingest(self, pulses):
        """
        Accepts a single bit or a list of bits and adds them to the queue.
        """
        if isinstance(pulses, list):
            for p in pulses:
                self.queue.append(p)
        else:
            self.queue.append(pulses)

    def pull(self):
        """
        Returns the oldest bit in the queue. 
        If queue is empty, returns 0 (Underflow).
        """
        if self.queue:
            return self.queue.popleft()
        return 0

    def query_system(self, n, clock_source, generator_source):
        """
        The Master Trigger: Forces the system to move and fill the queue.
        """
        stream = []
        while len(stream) < n:
            # 1. Trigger the Clocks and Generators
            clk_bit = clock_source.tick()
            gen_bits = generator_source.emit()
            
            # 2. Logic flows through modules...
            # (In the final assembly, this is where the cascade happens)
            
            # 3. Collect from the Emitter
            if self.queue:
                stream.append(self.pull())
            else:
                # Force a system cycle if the queue is dry
                pass 
        return stream

class LogicGate:
    def __init__(self, mode="XOR"):
        """
        mode: "AND", "OR", or "XOR"
        """
        self.mode = mode.upper()

    def process(self, input_a, input_b):
        """
        Performs the boolean operation on two input bits.
        """
        if self.mode == "AND":
            return 1 if (input_a == 1 and input_b == 1) else 0
        
        elif self.mode == "OR":
            return 1 if (input_a == 1 or input_b == 1) else 0
        
        elif self.mode == "XOR":
            return 1 if (input_a != input_b) else 0
        
        return 0

    def set_mode(self, new_mode):
        self.mode = new_mode.upper()

class ClockDivider:
    def __init__(self, division_factor=2):
        """
        division_factor: The ratio of input pulses to output pulses.
        2 = Divide by 2 (Half frequency: 1010 -> 1000)
        4 = Divide by 4 (Quarter frequency: 1111 -> 1000)
        """
        self.division_factor = max(1, division_factor)
        self.pulse_counter = 0

    def process(self, input_pulse):
        """
        Takes an input pulse and returns a divided output pulse.
        """
        if input_pulse == 1:
            self.pulse_counter += 1
            
            # Check if we have accumulated enough pulses to fire
            if self.pulse_counter >= self.division_factor:
                self.pulse_counter = 0
                return 1
        
        return 0

    def set_division(self, n):
        """Dynamically changes the clock speed ratio."""
        self.division_factor = max(1, n)
        self.pulse_counter = 0

class PulseGenerator:
    def __init__(self, pattern=[1], loop=True):
        """
        pattern: The sequence to pulse (e.g., [1, 0, 1, 1]).
        loop: If True, repeats the pattern indefinitely.
        """
        self.pattern = pattern
        self.loop = loop
        self.index = 0
        self.active = True

    def emit(self):
        """
        Returns the next bit in the sequence.
        """
        if not self.active:
            return 0
            
        bit = self.pattern[self.index]
        self.index += 1
        
        if self.index >= len(self.pattern):
            if self.loop:
                self.index = 0
            else:
                self.active = False
                
        return bit

    def trigger_reset(self):
        """Resets the pattern to the beginning."""
        self.index = 0
        self.active = True

class SystemController:
    """
    The Orchestrator: Wires modules together into a functional topology.
    """
    def __init__(self, nodes=48):
        # 1. Sources
        self.master_clock = ClockModule(interval=1)
        self.pulse_gen = PulseGenerator(pattern=[1, 0, 1, 1], loop=True)
        
        # 2. Regulation Layer
        self.divider = ClockDivider(division_factor=3)
        self.gate = ProbabilisticGate(pass_likelihood=0.9)
        
        # 3. Pattern Recognition Layer
        self.envelope = AccumulatingEnvelope(pattern=[1, 0, 1], n_outputs=nodes)
        
        # 4. The Complexity Core
        self.oscillator = ConfigurableOscillator(n=nodes)
        # Configure unique pulse widths for nodes to break linearity
        for i in range(nodes):
            self.oscillator.set_port_width(i, (i % 7) + 1)
            
        # 5. Output Management
        self.joiner = QueuedJoiner(capacity=2000000)

    def step(self):
        """
        One virtual cycle: Pulse -> Logic -> Oscillator -> Joiner
        """
        # A. Trigger Sources
        c_bit = self.master_clock.tick()
        p_bit = self.pulse_gen.emit()
        
        # B. Logic Processing
        # XOR the clock and pulse generator to create a jittered base signal
        base_signal = c_bit ^ p_bit
        
        # Pass through divider and probabilistic gate
        divided_signal = self.divider.process(base_signal)
        gated_signal = self.gate.process(divided_signal)
        
        # C. Envelope Pattern Match
        # If the gated signal matches the envelope pattern, it triggers 'N' nodes
        envelope_vector = self.envelope.receive(gated_signal)
        
        # D. Oscillator Update
        # The oscillator uses the envelope vector to flip its internal nodes
        current_state = self.oscillator.tick(envelope_vector)
        
        # E. Emission Selection
        # We select bit 0 of the state as our output for this cycle
        self.joiner.ingest(current_state[0])

    def generate(self, bit_count):
        """
        Queries the system for a specific amount of bits.
        """
        stream = []
        for _ in range(bit_count):
            self.step()
            stream.append(self.joiner.pull())
        return stream

# --- EXECUTION AND TESTING ---

if __name__ == "__main__":
    # 1. Initialize System
    engine = SystemController(nodes=48)
    
    # 2. Query for 1,000,000 bits
    print("[*] Powering up modules...")
    print("[*] Querying System for 1,000,000 bits...")
    bits = engine.generate(1000000)
    
    # 3. Quick Stats
    bit_str = "".join(map(str, bits[:100]))
    ones = sum(bits)
    print(f"[*] Sample: {bit_str}...")
    print(f"[*] Density: {ones/1000000:.4f} (Ideal: 0.5)")