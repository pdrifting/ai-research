# Bruteforcing Bias of the Sigmoid Discriminator to Solve Boolean Logic Gates

The core of this research was rooted in attempts to discover alternate sources for entropy in staved and barren environments.  This documents the complete pre‑alpha implementation demonstrating that Boolean logic can be represented, evaluated, and composed using continuous activation functions. The system in this paper uses brute‑forced weight and bias parameters for a sigmoid neuron (or small ensemble) to approximate Boolean truth tables.

### This implementation proves:
- Functional completeness (all Boolean gates implemented)
- Composability (gates can be stacked to form higher‑order logic)
- Evaluability (expressions can be parsed and executed)
- Deterministic behavior (outputs match truth tables)

This pre‑alpha engine is the computational ancestor of the alpha biological simulation.

## 1. Activation Function
```vb
Function Sigmoid(x As Single) As Single
    Return 1.0F / (1.0F + CSng(Math.Exp(-x)))
End Function
```

**Description**

A standard logistic sigmoid is used to approximate a hard threshold. Large‑magnitude weights (≈15–100) force the sigmoid into near‑binary behavior. This is the continuous precursor to the alpha engine’s soma potential threshold.

## 2. Primitive Boolean Gates (Single Neuron)

Each gate is implemented as:

- Output from calls to the Sigmoid Discriminator
```VB
output = sigmoid(w1*A + w2*B + b)
```

- AND
```vb
Function Gate_AND(A As Single, B As Single) As Single
    Dim z = A * 15.8151F + B * 15.8151F + -22.5764F
    Return Sigmoid(z)
End Function
```

- NAND
```vb
Function Gate_NAND(A As Single, B As Single) As Single
    Dim z = A * -15.8151F + B * -15.8151F + 22.5764F
    Return Sigmoid(z)
End Function
```

- OR
```vb
Function Gate_OR(A As Single, B As Single) As Single
    Dim z = A * 14.7613F + B * 14.7613F + -8.1472F
    Return Sigmoid(z)
End Function
```

- NOR
```vb
Function Gate_NOR(A As Single, B As Single) As Single
    Dim z = A * -14.7613F + B * -14.7613F + 8.1472F
    Return Sigmoid(z)
End Function
```

- NOT
```vb
Function Gate_NOT(x As Single) As Single
    Dim z = x * -100.0F + 16.6991F
    Return Sigmoid(z)
End Function
```

#### Interpretation
These weight sets were brute‑forced to match Boolean truth tables.

**They demonstrate:**
- symmetry (AND/OR families)
- sign inversion (NAND/NOR)
- extreme weights for NOT (hard inversion)

This is the mathematical precursor to ion‑flux heterogeneity in the alpha engine.

## 3. Composite Boolean Gates (Multi‑Neuron Ensembles)

XOR and XNOR require non‑linear composition.

- XOR
```vb
Function Gate_XOR(A As Single, B As Single) As Single
    Dim z0 = A * 15.0F + B * -16.0F + -6.4094F
    Dim z1 = A * 15.0F + B * 15.0F + -6.0F
    Dim z2 = A * -16.0F + B * 15.0F + -6.4094F
    Dim a0 = Sigmoid(z0)
    Dim a1 = Sigmoid(z1)
    Dim a2 = Sigmoid(z2)
    Return (a0 + a1 + a2) / 3.0F
End Function
```

- XNOR
```vb
Function Gate_XNOR(A As Single, B As Single) As Single
    Dim z0 = A * -15.0F + B * 16.0F + 6.4094F
    Dim z1 = A * -15.0F + B * -15.0F + 6.0F
    Dim z2 = A * 16.0F + B * -15.0F + 6.4094F
    Dim a0 = Sigmoid(z0)
    Dim a1 = Sigmoid(z1)
    Dim a2 = Sigmoid(z2)
    Return (a0 + a1 + a2) / 3.0F
End Function
```

#### Interpretation

These ensembles demonstrate:
- non‑linear separability
- structural composition
- symmetry between XOR and XNOR
- ensemble averaging as a stable output mechanism

This is the conceptual ancestor of multi‑dendrite logic in the alpha engine.

## 4. Derived Logic Gates (Stacked Composition)

These gates are built by composing primitive gates proving stackability, and .

- IMPLIES
```vb
Function Gate_IMPLIES(A As Single, B As Single) As Single
    Return Gate_OR(Gate_NOT(A), B)
End Function
```

- EQUALS (Alias to XNOR)
```vb
Function Gate_EQUALS(A As Single, B As Single) As Single
    Return Gate_XNOR(A, B)
End Function
```

- Greater‑Than
```vb
Function Gate_GT(A As Single, B As Single) As Single
    Return Gate_AND(A, Gate_NOT(B))
End Function
```

- Less‑Than
```vb
Function Gate_LT(A As Single, B As Single) As Single
    Return Gate_AND(Gate_NOT(A), B)
End Function
```

#### Interpretation

This proves:
- composability
- functional completeness
- higher‑order logic
- procedural evaluation

This is the final step before the alpha engine replaced static weights with temporal biological dynamics.

## 5. Expression Evaluator

A incomplete but functional primitive evaluator that demonstrates logic can be parsed and executed programmatically.

```vb
Function EvaluateExpression(expr As String, A As Single, B As Single) As Single
    ' Incomplete expression router — assumes input format like "AND(NOT(A), B)"
    expr = expr.Trim.ToUpper()

    If expr = "A" Then Return A
    If expr = "B" Then Return B
    If expr = "NOT(A)" Then Return Gate_NOT(A)
    If expr = "NOT(B)" Then Return Gate_NOT(B)

    If expr = "AND(A,B)" Then Return Gate_AND(A, B)
    If expr = "OR(A,B)" Then Return Gate_OR(A, B)
    If expr = "NAND(A,B)" Then Return Gate_NAND(A, B)
    If expr = "NOR(A,B)" Then Return Gate_NOR(A, B)
    If expr = "XOR(A,B)" Then Return Gate_XOR(A, B)
    If expr = "XNOR(A,B)" Then Return Gate_XNOR(A, B)
    If expr = "IMPLIES(A,B)" Then Return Gate_IMPLIES(A, B)
    If expr = "GT(A,B)" Then Return Gate_GT(A, B)
    If expr = "LT(A,B)" Then Return Gate_LT(A, B)
    If expr = "EQUALS(A,B)" Then Return Gate_EQUALS(A, B)

    Console.WriteLine($"Unknown expression: {expr}")
    Return -1.0F
End Function
```

This is mapping out the earliest form of a logic interpreter in the Ion Model lineage.

## 6. Full Gate Verification Harness

The test harness:
- enumerates all input combinations
- evaluates each gate
- thresholds outputs

```VB
Structure TestCase
    Dim A As Single
    Dim B As Single
End Structure

Function ToBinary(x As Single) As Integer
    Return If(x >= 0.5F, 1, 0)
End Function

Sub TestGate(name As String, gateFunc As Func(Of Single, Single, Single), expected() As Integer)
    Dim cases() As TestCase = {
        New TestCase With {.A = 0.0F, .B = 0.0F},
        New TestCase With {.A = 0.0F, .B = 1.0F},
        New TestCase With {.A = 1.0F, .B = 0.0F},
        New TestCase With {.A = 1.0F, .B = 1.0F}
    }

    Console.WriteLine($"--- Testing Gate: {name} ---")
    For i = 0 To cases.Length - 1
        Dim A = cases(i).A
        Dim B = cases(i).B
        Dim output = gateFunc(A, B)
        Dim predicted = ToBinary(output)
        Console.WriteLine($"Input ({A}, {B}) → Output: {predicted}  [Expected: {expected(i)}]")
    Next
    Console.WriteLine()
End Sub

Sub Main()
    ' Define truth tables (expected outputs) for each gate
    Dim truth_AND() = {0, 0, 0, 1}
    Dim truth_NAND() = {1, 1, 1, 0}
    Dim truth_OR() = {0, 1, 1, 1}
    Dim truth_NOR() = {1, 0, 0, 0}
    Dim truth_XOR() = {0, 1, 1, 0}
    Dim truth_XNOR() = {1, 0, 0, 1}
    Dim truth_IMPLIES() = {1, 1, 0, 1}
    Dim truth_EQUALS() = {1, 0, 0, 1}
    Dim truth_GT() = {0, 0, 1, 0}
    Dim truth_LT() = {0, 1, 0, 0}

    Console.WriteLine("=== Full Gate Verification ===" & vbLf)

    TestGate("AND", AddressOf Gate_AND, truth_AND)
    TestGate("NAND", AddressOf Gate_NAND, truth_NAND)
    TestGate("OR", AddressOf Gate_OR, truth_OR)
    TestGate("NOR", AddressOf Gate_NOR, truth_NOR)
    TestGate("XOR", AddressOf Gate_XOR, truth_XOR)
    TestGate("XNOR", AddressOf Gate_XNOR, truth_XNOR)
    TestGate("IMPLIES", AddressOf Gate_IMPLIES, truth_IMPLIES)
    TestGate("EQUALS", AddressOf Gate_EQUALS, truth_EQUALS)
    TestGate("GT (Greater Than)", AddressOf Gate_GT, truth_GT)
    TestGate("LT (Less Than)", AddressOf Gate_LT, truth_LT)

    Console.WriteLine("=== Testing Single Input Gate: ===")
    For Each x In New Single() {0.0F, 1.0F}
        Dim output = Gate_NOT(x)
        Console.WriteLine($"NOT({x}) = {ToBinary(output)} [Expected: {If(x = 0.0F, 1, 0)}]")
    Next
    Console.WriteLine()
End Sub
```

<pre>
=== Full Gate Verification ===

--- Testing Gate: AND ---
Input (0, 0) → Output: 0  [Expected: 0]
Input (0, 1) → Output: 0  [Expected: 0]
Input (1, 0) → Output: 0  [Expected: 0]
Input (1, 1) → Output: 1  [Expected: 1]

--- Testing Gate: NAND ---
Input (0, 0) → Output: 1  [Expected: 1]
Input (0, 1) → Output: 1  [Expected: 1]
Input (1, 0) → Output: 1  [Expected: 1]
Input (1, 1) → Output: 0  [Expected: 0]

--- Testing Gate: OR ---
Input (0, 0) → Output: 0  [Expected: 0]
Input (0, 1) → Output: 1  [Expected: 1]
Input (1, 0) → Output: 1  [Expected: 1]
Input (1, 1) → Output: 1  [Expected: 1]

--- Testing Gate: NOR ---
Input (0, 0) → Output: 1  [Expected: 1]
Input (0, 1) → Output: 0  [Expected: 0]
Input (1, 0) → Output: 0  [Expected: 0]
Input (1, 1) → Output: 0  [Expected: 0]

--- Testing Gate: XOR ---
Input (0, 0) → Output: 0  [Expected: 0]
Input (0, 1) → Output: 1  [Expected: 1]
Input (1, 0) → Output: 1  [Expected: 1]
Input (1, 1) → Output: 0  [Expected: 0]

--- Testing Gate: XNOR ---
Input (0, 0) → Output: 1  [Expected: 1]
Input (0, 1) → Output: 0  [Expected: 0]
Input (1, 0) → Output: 0  [Expected: 0]
Input (1, 1) → Output: 1  [Expected: 1]

--- Testing Gate: IMPLIES ---
Input (0, 0) → Output: 1  [Expected: 1]
Input (0, 1) → Output: 1  [Expected: 1]
Input (1, 0) → Output: 0  [Expected: 0]
Input (1, 1) → Output: 1  [Expected: 1]

--- Testing Gate: EQUALS ---
Input (0, 0) → Output: 1  [Expected: 1]
Input (0, 1) → Output: 0  [Expected: 0]
Input (1, 0) → Output: 0  [Expected: 0]
Input (1, 1) → Output: 1  [Expected: 1]

--- Testing Gate: GT (Greater Than) ---
Input (0, 0) → Output: 0  [Expected: 0]
Input (0, 1) → Output: 0  [Expected: 0]
Input (1, 0) → Output: 1  [Expected: 1]
Input (1, 1) → Output: 0  [Expected: 0]

--- Testing Gate: LT (Less Than) ---
Input (0, 0) → Output: 0  [Expected: 0]
Input (0, 1) → Output: 1  [Expected: 1]
Input (1, 0) → Output: 0  [Expected: 0]
Input (1, 1) → Output: 0  [Expected: 0]

=== Testing Single Input Gate: ===
NOT(0) = 1 [Expected: 1]
NOT(1) = 0 [Expected: 0]
</pre>

This is the formal proof that the brute‑forced weights behave as repeatable Boolean logic.

## 7. Failure of Gradient‑Based Learning to Discover Boolean Gate Weights

This section documents the empirical and structural reasons why PyTorch‑based training, using standard forward/backpropagation and smooth activation functions, fails to reliably discover the weight/bias configurations required to implement all the Boolean logic gates.

This failure was demonstrated repeatedly across dozens of dated research experiments in the src/ directory, using:
- PyTorch MLPs
- logistic sigmoid activations
- BCE loss
- Adam/SGD optimizers
- randomized initializations
- grid searches
- curriculum training
- noise injection
- annealing schedules

Despite extensive attempts, gradient descent could not converge to the brute‑forced solutions that were later found manually.  This failure is not incidental, it is a structural flaw in current AI model designs.

## 7.1 Boolean Gates Require Extremely Sharp Decision Boundaries

The brute‑forced weights for AND, OR, NAND, NOR, and NOT all share a key property:
- very large magnitude weights (≈ 15–100)
- biases tuned to narrow activation windows
- sigmoid outputs forced into near‑binary regions

These create quasi‑step‑functions inside a smooth activation function.

Gradient descent cannot reliably reach these regions because:
- the sigmoid saturates
- gradients vanish
- updates become negligible
- the optimizer cannot “push” weights into the required extremes

This is a known limitation of smooth activations attempting to approximate discrete logic.

## 7.2 XOR/XNOR Require Non‑Linear Ensembles With Precise Symmetry

The brute‑forced XOR/XNOR solutions require:
- three neurons
- symmetrical weight patterns
- precise sign inversions
- biases tuned to narrow regions
- ensemble averaging

PyTorch and Custom training arragements consistently failed to:
- discover the symmetry
- maintain sign‑inverted pairs
- stabilize the ensemble
- avoid collapsing into trivial solutions (always‑0 or always‑1)

Even with:
- multiple restarts
- high learning rates
- low learning rates
- weight decay
- no weight decay
- noise injection
- curriculum learning

This prevents traditional and custom modals based on known model designs could not reliably converge.

## 7.3 Gradient Descent Prefers Smooth, Probabilistic Solutions

Boolean gates require hard, discrete boundaries.

Gradient descent prefers:
- smooth transitions
- probabilistic outputs
- minimal‑norm solutions
- low‑magnitude weights
- wide basins of attraction

The brute‑forced Boolean weights sit in narrow, steep, high‑curvature regions of the loss landscape.

These regions:
- are difficult to reach
- are unstable
- have vanishing gradients
- are surrounded by flat plateaus

PyTorch optimizers simply do not explore these regions naturally.

## 7.4 The Loss Landscape Contains “Dead Zones” Around Boolean Solutions

Experiments showed that:
- near the correct Boolean weights, gradients approach zero
- the optimizer stalls before reaching the required magnitude
- the loss surface becomes nearly flat
- the model cannot “snap” into the discrete regime

This is why brute‑forcing succeeded where training failed:
- brute force explores the entire space
- gradient descent only explores local slopes
- Boolean logic requires global search, not local optimization.

## 7.5 PyTorch Training Collapses Under Binary Truth‑Table Targets

When training on truth tables:
- the model often converges to trivial solutions
- OR collapses to always‑1
- AND collapses to always‑0
- XOR collapses to 0.5 everywhere
- NOT collapses to 0.5 for both inputs

This is because:
- the loss surface has wide, shallow minima
- the correct Boolean minima are narrow and steep
- gradients vanish before reaching them

This was confirmed across:
- MLPs
- single‑neuron models
- multi‑layer models
- ensembles
- different optimizers

None reliably converged.

## 7.6 Summary of Why PyTorch Cannot Learn These Gates Reliably

<ol type="a">
  <li>Sigmoid saturation prevents reaching extreme weights.</li>
  <li>Vanishing gradients block movement toward discrete boundaries.</li>
  <li>Loss landscape geometry traps the optimizer in trivial minima.</li>
  <li>Symmetry requirements for XOR/XNOR are not naturally discovered.</li>
  <li>High‑curvature regions are inaccessible to gradient descent.</li>
  <li>Discrete logic is fundamentally misaligned with smooth optimization.</li>
  <li>Binary truth tables create unstable training dynamics.</li>
</ol>

## 7.7 Inability of a Single Model to Represent All Boolean Gates Simultaneously

An additional structural limitation became clear during the pre‑alpha research experiments (documented in source files throughout the src/ directory), that even when a PyTorch model successfully learned one Boolean gate, no configuration was found that could represent all Boolean gates within a single shared parameter space.

This limitation was not merely empirical, it is was rooted in the geometry of the activation function and the nature of the Boolean gates themselves.

#### 7.7.1 Boolean Gates Occupy Mutually Exclusive Activation Regimes

The brute‑forced solutions for AND, OR, NAND, NOR, and NOT all require:
- extremely large positive weights
- extremely large negative weights
- sharply tuned biases
- near‑binary sigmoid saturation

But critically:
- Each gate requires a different extreme region of parameter space.
  For example:
    - AND requires large positive weights and a strong negative bias
    - NAND requires large negative weights and a strong positive bias
    - OR requires moderate positive weights and a less negative bias
    - NOR requires moderate negative weights and a less positive bias
    - NOT requires extreme negative weight and a large positive bias

These regions are:
- far apart
- narrow
- high‑curvature
- separated by flat plateaus
- incompatible with one another

A single set of weights cannot simultaneously satisfy all these constraints.

#### 7.7.2 Shared‑Parameter Models Cannot Express Opposing Logic Boundaries

In a traditional neural network:
- all gates share the same learned parameters
- the model must represent all functions with the same weights
- the activation function is smooth and continuous

But Boolean gates require opposing decision boundaries:
- AND and NAND are exact sign inverses
- OR and NOR are exact sign inverses
- NOT requires a boundary orthogonal to both inputs
- XOR/XNOR require multi‑neuron ensembles with symmetry constraints

These boundaries cannot coexist in a single sigmoid neuron or a single MLP without:
- catastrophic interference
- collapse to trivial solutions
- loss of separability
- gradient cancellation
- This was observed repeatedly in the experiments.
  
#### 7.7.3 No Known Architecture Can Represent All Boolean Gates in One Model

Across all experiments:
- single‑layer networks failed
- multi‑layer networks failed
- wide networks failed
- deep networks failed
- ensembles failed
- residual connections failed
- attention‑style gating failed
- curriculum learning failed
- noise‑based exploration failed

Even when a model successfully learned one gate (e.g., AND), it could not:
- retain that solution
- simultaneously represent OR
- simultaneously represent NAND
- simultaneously represent XOR
- or generalize across all gates

The model would collapse into:
- always‑0
- always‑1
- probabilistic outputs
- or unstable oscillations

This strongly suggests that the Boolean gates occupy mutually exclusive attractor basins in the parameter space of smooth neural networks.

#### 7.7.4 Conjecture: Boolean Gates Require Distinct, Non‑Coexisting Activation States

Based on the empirical evidence, each Boolean gate corresponds to a distinct activation regime that cannot coexist with the others in a single continuous model.

This is because:
- the required weight magnitudes differ by sign and scale
- the required biases differ by sign and magnitude
- the sigmoid must saturate in different regions for each gate
- the loss minima for each gate are isolated and incompatible
- Thus, a single model cannot simultaneously satisfy:
  - AND’s decision boundary
  - OR’s decision boundary
  - NAND’s inverted boundary
  - NOR’s inverted boundary
  - NOT’s extreme inversion boundary
  - XOR/XNOR’s multi‑neuron symmetry constraints
  
This is not a training failure, it is a representational impossibility under standard known architectures.

#### 7.7.5 Why This Matters for the Alpha Engine

This limitation directly motivated the transition from:
<div><b>static weight‑based logic</b></div>
<div><i>to</i></div>
<div><b>temporal, biological, ion‑flux‑driven computation</b></div>
<br>

The alpha engine solves this by:
- abandoning shared weights
- abandoning smooth activations
- abandoning gradient descent
- using heterogeneous synaptic flux
- using temporal integration
- using state‑dependent dynamics
- using structural logic encoding

Instead of trying to force all gates into one weight space, the alpha engine encodes logic in the physical configuration of synapses and dendrites, not in shared parameters.

This is the conceptual breakthrough that makes the alpha engine possible.
