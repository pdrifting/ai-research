from typing import Callable, Dict, List, Tuple
import uuid, os, pickle

# -----------------------------
# Node Types
# -----------------------------
class NodeType:
    FUNCTION = "FUNCTION"
    GATE = "GATE"

# -----------------------------
# Graph Node
# -----------------------------
class GraphNode:
    def __init__(self, node_type: str, label: str, func: Callable = None):
        self.id = str(uuid.uuid4())
        self.type = node_type
        self.label = label
        self.func = func
        self.inputs: List[str] = []

# -----------------------------
# Component Graph
# -----------------------------
class ComponentGraph:
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}

    def add_node(self, node: GraphNode):
        self.nodes[node.id] = node

    def connect(self, from_id: str, to_id: str):
        self.nodes[to_id].inputs.append(from_id)

    def simulate(self, input_values: Dict[str, bool]) -> Dict[str, bool]:
        results: Dict[str, bool] = {}

        def evaluate(node_id: str) -> bool:
            if node_id in results:
                return results[node_id]

            node = self.nodes[node_id]
            if node.type == NodeType.FUNCTION and node.func is None:
                results[node_id] = input_values.get(node_id, False)
                return results[node_id]

            input_vals = [evaluate(inp_id) for inp_id in node.inputs]
            result = node.func(*input_vals)
            results[node_id] = result
            return result

        for node_id in self.nodes:
            evaluate(node_id)

        return results

# -----------------------------
# Boolean Gate Functions
# -----------------------------
def and_gate(a: bool, b: bool) -> bool: return a and b
def or_gate(a: bool, b: bool) -> bool: return a or b
def not_gate(a: bool) -> bool: return not a
def nand(a: bool, b: bool) -> bool: return not (a and b)
def nor(a: bool, b: bool) -> bool: return not (a or b)
def xor(a: bool, b: bool) -> bool: return a != b
def xnor(a: bool, b: bool) -> bool: return a == b

# -----------------------------
# Composite Logic Functions
# -----------------------------
def half_adder(a: bool, b: bool) -> Tuple[bool, bool]:
    return xor(a, b), and_gate(a, b)

def full_adder(a: bool, b: bool, carry_in: bool) -> Tuple[bool, bool]:
    sum1, carry1 = half_adder(a, b)
    sum2, carry2 = half_adder(sum1, carry_in)
    return sum2, or_gate(carry1, carry2)

def mux2to1(a: bool, b: bool, sel: bool) -> bool:
    return (a and not_gate(sel)) or (b and sel)

def demux1to2(input_bit: bool, sel: bool) -> Tuple[bool, bool]:
    return (input_bit if not sel else False, input_bit if sel else False)

# -----------------------------
# Flip-Flops
# -----------------------------
class SRFlipFlop:
    def __init__(self): self.q = False
    def update(self, s: bool, r: bool) -> bool:
        if s and not r: self.q = True
        elif r and not s: self.q = False
        return self.q

class DFlipFlop:
    def __init__(self): self.q = False
    def update(self, d: bool, clk: bool) -> bool:
        if clk: self.q = d
        return self.q

# -----------------------------
# Basic Gate Functions
# -----------------------------
def and_gate(*args):
    return all(args)

def or_gate(*args):
    return any(args)

def not_gate(x):
    return not x

def nand_gate(*args):
    return not all(args)

def nor_gate(*args):
    return not any(args)

def xor(a, b):
    return a != b

def xnor(a, b):
    return a == b

# -----------------------------
# Flip-Flop Wrappers (Pickle-safe)
# -----------------------------
def sr_flipflop_func(s: bool, r: bool) -> bool:
    ff = SRFlipFlop()
    return ff.update(s, r)

def d_flipflop_func(d: bool, clk: bool) -> bool:
    ff = DFlipFlop()
    return ff.update(d, clk)

# -----------------------------
# Truth Tables
# -----------------------------
truth_tables = {
    "AND": [((False, False), False), ((False, True), False),
            ((True, False), False), ((True, True), True)],
    "OR": [((False, False), False), ((False, True), True),
           ((True, False), True), ((True, True), True)],
    "NOT": [((False,), True), ((True,), False)],
    "NAND": [((False, False), True), ((False, True), True),
             ((True, False), True), ((True, True), False)],
    "NOR": [((False, False), True), ((False, True), False),
            ((True, False), False), ((True, True), False)],
    "XOR": [((False, False), False), ((False, True), True),
            ((True, False), True), ((True, True), False)],
    "XNOR": [((False, False), True), ((False, True), False),
             ((True, False), False), ((True, True), True)],
}

# -----------------------------
# Graph Persistence
# -----------------------------
def save_graph(graph: ComponentGraph, name: str):
    os.makedirs("learned_graphs", exist_ok=True)
    with open(f"learned_graphs/{name}.graph", "wb") as f:
        pickle.dump(graph, f)

def load_graph(name: str) -> ComponentGraph:
    path = f"learned_graphs/{name}.graph"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# -----------------------------
# Graph Visualizer
# -----------------------------
def visualize_graph(graph: ComponentGraph, title: str):
    print(f"\n📊 [GRAPH: {title}]")
    for node_id, node in graph.nodes.items():
        input_labels = [graph.nodes[inp_id].label for inp_id in node.inputs]
        print(f"• {node.label} ({node.type}) ← {input_labels}")

# -----------------------------
# Gate Learning & Validation
# -----------------------------
def learn_gate(gate_label: str, symbolic_func: Callable) -> ComponentGraph:
    graph = ComponentGraph()

    gate_node = GraphNode(NodeType.GATE, gate_label, symbolic_func)
    graph.add_node(gate_node)

    input_nodes = []
    for i in range(len(truth_tables[gate_label][0][0])):
        input_node = GraphNode(NodeType.FUNCTION, f"{gate_label}_IN{i}")
        graph.add_node(input_node)
        graph.connect(input_node.id, gate_node.id)
        input_nodes.append(input_node)

    passed = True
    for inputs, expected in truth_tables[gate_label]:
        input_map = {input_nodes[i].id: val for i, val in enumerate(inputs)}
        result = graph.simulate(input_map).get(gate_node.id)
        if result != expected:
            passed = False
            break

    if passed:
        print(f"[LEARNED] {gate_label} passed truth table.")
        gate_node.func = symbolic_func
        gate_node.type = NodeType.FUNCTION
    else:
        print(f"[FAILED] {gate_label} did not match truth table.")

    return graph

def validate_gate(graph: ComponentGraph, gate_label: str):
    for node in graph.nodes.values():
        if node.label == gate_label and node.type == NodeType.FUNCTION:
            table = truth_tables.get(gate_label)
            print(f"\n[VALIDATING] {gate_label}")
            for inputs, expected in table:
                input_map = {node.inputs[i]: val for i, val in enumerate(inputs)}
                result = graph.simulate(input_map).get(node.id)
                print(f"{gate_label} {inputs} → {result} (Expected: {expected})")

# -----------------------------
# Composite Graph Builders
# -----------------------------
def build_half_adder_graph() -> ComponentGraph:
    graph = ComponentGraph()
    a = GraphNode(NodeType.FUNCTION, "HA_A")
    b = GraphNode(NodeType.FUNCTION, "HA_B")
    graph.add_node(a)
    graph.add_node(b)

    xor_node = GraphNode(NodeType.FUNCTION, "HalfAdder_SUM", xor)
    and_node = GraphNode(NodeType.FUNCTION, "HalfAdder_CARRY", and_gate)
    graph.add_node(xor_node)
    graph.add_node(and_node)

    graph.connect(a.id, xor_node.id)
    graph.connect(b.id, xor_node.id)
    graph.connect(a.id, and_node.id)
    graph.connect(b.id, and_node.id)

    return graph

def build_full_adder_graph() -> ComponentGraph:
    graph = ComponentGraph()
    a = GraphNode(NodeType.FUNCTION, "FA_A")
    b = GraphNode(NodeType.FUNCTION, "FA_B")
    cin = GraphNode(NodeType.FUNCTION, "FA_CIN")
    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(cin)

    xor1 = GraphNode(NodeType.FUNCTION, "XOR1", xor)
    xor2 = GraphNode(NodeType.FUNCTION, "FullAdder_SUM", xor)
    and1 = GraphNode(NodeType.FUNCTION, "AND1", and_gate)
    and2 = GraphNode(NodeType.FUNCTION, "AND2", and_gate)
    or_node = GraphNode(NodeType.FUNCTION, "FullAdder_CARRY", or_gate)

    graph.add_node(xor1)
    graph.add_node(xor2)
    graph.add_node(and1)
    graph.add_node(and2)
    graph.add_node(or_node)

    graph.connect(a.id, xor1.id)
    graph.connect(b.id, xor1.id)
    graph.connect(xor1.id, xor2.id)
    graph.connect(cin.id, xor2.id)

    graph.connect(a.id, and1.id)
    graph.connect(b.id, and1.id)
    graph.connect(xor1.id, and2.id)
    graph.connect(cin.id, and2.id)

    graph.connect(and1.id, or_node.id)
    graph.connect(and2.id, or_node.id)

    return graph

def build_mux_graph() -> ComponentGraph:
    graph = ComponentGraph()
    a = GraphNode(NodeType.FUNCTION, "MUX_A")
    b = GraphNode(NodeType.FUNCTION, "MUX_B")
    sel = GraphNode(NodeType.FUNCTION, "MUX_SEL")
    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(sel)

    not_sel = GraphNode(NodeType.FUNCTION, "NOT_SEL", not_gate)
    and1 = GraphNode(NodeType.FUNCTION, "AND1", and_gate)
    and2 = GraphNode(NodeType.FUNCTION, "AND2", and_gate)
    or_node = GraphNode(NodeType.FUNCTION, "MUX_OUT", or_gate)

    graph.add_node(not_sel)
    graph.add_node(and1)
    graph.add_node(and2)
    graph.add_node(or_node)

    graph.connect(sel.id, not_sel.id)
    graph.connect(a.id, and1.id)
    graph.connect(not_sel.id, and1.id)
    graph.connect(b.id, and2.id)
    graph.connect(sel.id, and2.id)
    graph.connect(and1.id, or_node.id)
    graph.connect(and2.id, or_node.id)

    return graph

def build_demux_graph() -> ComponentGraph:
    graph = ComponentGraph()
    inp = GraphNode(NodeType.FUNCTION, "DEMUX_IN")
    sel = GraphNode(NodeType.FUNCTION, "DEMUX_SEL")
    graph.add_node(inp)
    graph.add_node(sel)

    not_sel = GraphNode(NodeType.FUNCTION, "NOT_SEL", not_gate)
    out0 = GraphNode(NodeType.FUNCTION, "DEMUX_OUT0", and_gate)
    out1 = GraphNode(NodeType.FUNCTION, "DEMUX_OUT1", and_gate)

    graph.add_node(not_sel)
    graph.add_node(out0)
    graph.add_node(out1)

    graph.connect(sel.id, not_sel.id)
    graph.connect(inp.id, out0.id)
    graph.connect(not_sel.id, out0.id)
    graph.connect(inp.id, out1.id)
    graph.connect(sel.id, out1.id)

    return graph

def build_sr_flipflop_graph() -> ComponentGraph:
    graph = ComponentGraph()
    s = GraphNode(NodeType.FUNCTION, "SR_S")
    r = GraphNode(NodeType.FUNCTION, "SR_R")
    graph.add_node(s)
    graph.add_node(r)

    sr_ff = GraphNode(NodeType.FUNCTION, "SRFlipFlop_OUT", sr_flipflop_func)
    graph.add_node(sr_ff)
    graph.connect(s.id, sr_ff.id)
    graph.connect(r.id, sr_ff.id)

    return graph

def build_d_flipflop_graph() -> ComponentGraph:
    graph = ComponentGraph()
    d = GraphNode(NodeType.FUNCTION, "DFF_D")
    clk = GraphNode(NodeType.FUNCTION, "DFF_CLK")
    graph.add_node(d)
    graph.add_node(clk)

    d_ff = GraphNode(NodeType.FUNCTION, "DFlipFlop_OUT", d_flipflop_func)
    graph.add_node(d_ff)
    graph.connect(d.id, d_ff.id)
    graph.connect(clk.id, d_ff.id)

    return graph

if __name__ == "__main__":
    # Learn and validate basic gates
    for gate_label, func in [("AND", and_gate), ("OR", or_gate), ("NOT", not_gate),
                             ("NAND", nand_gate), ("NOR", nor_gate),
                             ("XOR", xor), ("XNOR", xnor)]:
        graph = learn_gate(gate_label, func)
        save_graph(graph, gate_label)
        validate_gate(graph, gate_label)
        visualize_graph(graph, gate_label)

    # Build and visualize composite circuits
    circuits = {
        "HalfAdder": build_half_adder_graph,
        "FullAdder": build_full_adder_graph,
        "MUX": build_mux_graph,
        "DEMUX": build_demux_graph,
        "SRFlipFlop": build_sr_flipflop_graph,
        "DFlipFlop": build_d_flipflop_graph,
    }

    for label, builder in circuits.items():
        graph = builder()
        save_graph(graph, label)
        visualize_graph(graph, label)

        # Simulate with sample inputs
        print(f"\n🔬 [SIMULATING] {label}")
        sample_inputs = {
            "HalfAdder": {"HA_A": True, "HA_B": False},
            "FullAdder": {"FA_A": True, "FA_B": True, "FA_CIN": True},
            "MUX": {"MUX_A": False, "MUX_B": True, "MUX_SEL": True},
            "DEMUX": {"DEMUX_IN": True, "DEMUX_SEL": False},
            "SRFlipFlop": {"SR_S": True, "SR_R": False},
            "DFlipFlop": {"DFF_D": True, "DFF_CLK": True},
        }
        inputs = sample_inputs.get(label, {})
        result = graph.simulate(inputs)
        for node_id, value in result.items():
            print(f"{graph.nodes[node_id].label}: {value}")