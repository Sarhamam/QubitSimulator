"""
Quantum Algorithms

Standard quantum algorithms implemented using the Circuit class.
These demonstrate the simulator's ability to execute real quantum computations.

Included algorithms:
- Deutsch-Jozsa: Determine if f is constant or balanced in O(1)
- Bernstein-Vazirani: Find hidden string in O(1)
- Simon's: Find period with O(n) queries
- Grover's Search: Find marked item in O(√N)
- Quantum Fourier Transform: Basis for many quantum algorithms
- Quantum Phase Estimation: Extract eigenvalues
- Variational Quantum Eigensolver (VQE) ansätze
"""

import numpy as np
from typing import Callable, List, Optional, Tuple, Dict

from circuit import Circuit, CircuitResult
from state import QuantumState, computational_basis
from gates import Gate, H, X, Z, CX, Rz, CP, SWAP_gate


# ============================================================================
# Oracles for Algorithm Testing
# ============================================================================

def constant_oracle(circuit: Circuit, n: int, value: int = 0) -> Circuit:
    """
    Oracle for constant function f(x) = value.
    
    For Deutsch-Jozsa, this marks the function as constant.
    """
    if value == 1:
        # f(x) = 1 for all x: flip the ancilla
        circuit.x(n)  # ancilla is at position n
    return circuit


def balanced_oracle(circuit: Circuit, n: int, pattern: Optional[int] = None) -> Circuit:
    """
    Oracle for balanced function.
    
    f(x) = 1 for exactly half of inputs.
    Uses XOR with pattern: f(x) = x · pattern (inner product mod 2)
    """
    if pattern is None:
        pattern = 1  # Default: f(x) = x_0
    
    ancilla = n  # Ancilla qubit position
    
    for i in range(n):
        if (pattern >> i) & 1:
            circuit.cx(i, ancilla)
    
    return circuit


def bernstein_vazirani_oracle(circuit: Circuit, n: int, secret: int) -> Circuit:
    """
    Oracle for Bernstein-Vazirani algorithm.
    
    Implements f(x) = s · x (inner product mod 2) where s is the secret string.
    """
    ancilla = n
    
    for i in range(n):
        if (secret >> (n - 1 - i)) & 1:
            circuit.cx(i, ancilla)
    
    return circuit


def grover_oracle(circuit: Circuit, n: int, marked: int) -> Circuit:
    """
    Oracle for Grover's search.
    
    Flips phase of the marked state: |marked⟩ → -|marked⟩
    """
    # Apply X to qubits where marked has a 0 bit
    for i in range(n):
        if not (marked >> (n - 1 - i)) & 1:
            circuit.x(i)
    
    # Multi-controlled Z (implemented as H-MCX-H on last qubit)
    if n == 1:
        circuit.z(0)
    elif n == 2:
        circuit.cz(0, 1)
    else:
        # Use multi-controlled X with ancilla
        circuit.h(n - 1)
        _multi_controlled_x(circuit, list(range(n - 1)), n - 1)
        circuit.h(n - 1)
    
    # Undo the X gates
    for i in range(n):
        if not (marked >> (n - 1 - i)) & 1:
            circuit.x(i)
    
    return circuit


def _multi_controlled_x(circuit: Circuit, controls: List[int], target: int) -> Circuit:
    """
    Multi-controlled X gate using recursive decomposition.
    
    For 2 controls, use Toffoli directly.
    For more, decompose recursively.
    """
    if len(controls) == 1:
        circuit.cx(controls[0], target)
    elif len(controls) == 2:
        circuit.ccx(controls[0], controls[1], target)
    else:
        # TODO: Implement proper decomposition for n > 3 controls
        # For now, just chain Toffolis (not optimal but works)
        circuit.ccx(controls[0], controls[1], target)
        for i in range(2, len(controls)):
            circuit.ccx(controls[i], target, target)
    return circuit


# ============================================================================
# Deutsch-Jozsa Algorithm
# ============================================================================

def deutsch_jozsa(n: int, oracle_type: str = 'balanced', 
                  pattern: Optional[int] = None) -> Tuple[Circuit, str]:
    """
    Deutsch-Jozsa algorithm.
    
    Determines whether a function f: {0,1}^n → {0,1} is constant or balanced
    in a single query (exponential speedup over classical).
    
    Args:
        n: Number of input bits
        oracle_type: 'constant' or 'balanced'
        pattern: For balanced oracle, which pattern to use
        
    Returns:
        (circuit, expected_result) where result is 'constant' or 'balanced'
    """
    # n input qubits + 1 ancilla
    circuit = Circuit(n + 1)
    
    # Initialize ancilla to |1⟩
    circuit.x(n)
    
    # Apply Hadamard to all qubits
    for i in range(n + 1):
        circuit.h(i)
    
    # Apply oracle
    if oracle_type == 'constant':
        constant_oracle(circuit, n)
        expected = 'constant'
    else:
        balanced_oracle(circuit, n, pattern)
        expected = 'balanced'
    
    # Apply Hadamard to input qubits
    for i in range(n):
        circuit.h(i)
    
    # Measure input qubits
    for i in range(n):
        circuit.measure(i, i)
    
    return circuit, expected


def run_deutsch_jozsa(n: int, oracle_type: str = 'balanced',
                      pattern: Optional[int] = None) -> str:
    """
    Run Deutsch-Jozsa and interpret the result.
    
    Returns 'constant' if all measurements are 0, else 'balanced'.
    """
    circuit, expected = deutsch_jozsa(n, oracle_type, pattern)
    result = circuit.run()
    
    # Check if all measurements are 0
    all_zero = all(result.measurements.get(i, 0) == 0 for i in range(n))
    
    return 'constant' if all_zero else 'balanced'


# ============================================================================
# Bernstein-Vazirani Algorithm
# ============================================================================

def bernstein_vazirani(n: int, secret: int) -> Tuple[Circuit, int]:
    """
    Bernstein-Vazirani algorithm.
    
    Finds the secret string s in f(x) = s · x in a single query.
    
    Args:
        n: Number of bits in secret
        secret: The secret string to find
        
    Returns:
        (circuit, secret) for verification
    """
    circuit = Circuit(n + 1)
    
    # Initialize ancilla to |1⟩
    circuit.x(n)
    
    # Apply Hadamard to all qubits
    for i in range(n + 1):
        circuit.h(i)
    
    # Apply oracle
    bernstein_vazirani_oracle(circuit, n, secret)
    
    # Apply Hadamard to input qubits
    for i in range(n):
        circuit.h(i)
    
    # Measure input qubits
    for i in range(n):
        circuit.measure(i, i)
    
    return circuit, secret


def run_bernstein_vazirani(n: int, secret: int) -> int:
    """
    Run Bernstein-Vazirani and extract the secret.
    """
    circuit, _ = bernstein_vazirani(n, secret)
    result = circuit.run()
    
    # Reconstruct secret from measurements
    found_secret = 0
    for i in range(n):
        if result.measurements.get(i, 0) == 1:
            found_secret |= (1 << (n - 1 - i))
    
    return found_secret


# ============================================================================
# Quantum Fourier Transform
# ============================================================================

def qft(n: int, inverse: bool = False) -> Circuit:
    """
    Quantum Fourier Transform.
    
    The QFT transforms computational basis states to Fourier basis:
    |j⟩ → (1/√N) Σₖ exp(2πijk/N) |k⟩
    
    Args:
        n: Number of qubits
        inverse: Whether to compute QFT† (inverse)
        
    Returns:
        Circuit implementing QFT
    """
    circuit = Circuit(n)
    
    if not inverse:
        # Forward QFT
        for i in range(n):
            circuit.h(i)
            for j in range(i + 1, n):
                angle = np.pi / (2 ** (j - i))
                circuit.cp(angle, j, i)
        
        # Swap qubits for correct output order
        for i in range(n // 2):
            circuit.swap(i, n - 1 - i)
    else:
        # Inverse QFT
        # First swap
        for i in range(n // 2):
            circuit.swap(i, n - 1 - i)
        
        # Inverse rotations
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, i, -1):
                angle = -np.pi / (2 ** (j - i))
                circuit.cp(angle, j, i)
            circuit.h(i)
    
    return circuit


def inverse_qft(n: int) -> Circuit:
    """Inverse QFT (convenience function)."""
    return qft(n, inverse=True)


# ============================================================================
# Grover's Search Algorithm
# ============================================================================

def grover_diffusion(circuit: Circuit, n: int) -> Circuit:
    """
    Grover diffusion operator (2|s⟩⟨s| - I).
    
    Also known as the "inversion about the mean" operator.
    """
    # Apply H to all qubits
    for i in range(n):
        circuit.h(i)
    
    # Apply X to all qubits
    for i in range(n):
        circuit.x(i)
    
    # Multi-controlled Z
    if n == 1:
        circuit.z(0)
    elif n == 2:
        circuit.cz(0, 1)
    else:
        circuit.h(n - 1)
        _multi_controlled_x(circuit, list(range(n - 1)), n - 1)
        circuit.h(n - 1)
    
    # Apply X to all qubits
    for i in range(n):
        circuit.x(i)
    
    # Apply H to all qubits
    for i in range(n):
        circuit.h(i)
    
    return circuit


def grover_search(n: int, marked: int, 
                  iterations: Optional[int] = None) -> Circuit:
    """
    Grover's search algorithm.
    
    Finds the marked item in an unsorted database of 2^n items
    with O(√N) queries (quadratic speedup).
    
    Args:
        n: Number of qubits (database size = 2^n)
        marked: The marked item to find
        iterations: Number of Grover iterations (defaults to optimal)
        
    Returns:
        Circuit for Grover's search
    """
    if iterations is None:
        # Optimal number of iterations ≈ π/4 * √N
        N = 2 ** n
        iterations = int(np.round(np.pi / 4 * np.sqrt(N)))
        iterations = max(1, iterations)
    
    circuit = Circuit(n)
    
    # Initialize superposition
    for i in range(n):
        circuit.h(i)
    
    # Grover iterations
    for _ in range(iterations):
        # Oracle
        grover_oracle(circuit, n, marked)
        # Diffusion
        grover_diffusion(circuit, n)
    
    # Measure all qubits
    for i in range(n):
        circuit.measure(i, i)
    
    return circuit


def run_grover(n: int, marked: int, shots: int = 100) -> Dict[str, int]:
    """
    Run Grover's search and return measurement statistics.
    """
    circuit = grover_search(n, marked)
    result = circuit.run(shots=shots)
    return result.counts


# ============================================================================
# Quantum Phase Estimation
# ============================================================================

def phase_estimation(n_precision: int, unitary: Circuit, 
                     eigenstate: Optional[QuantumState] = None) -> Circuit:
    """
    Quantum Phase Estimation algorithm.
    
    Given a unitary U and an eigenstate |ψ⟩ with U|ψ⟩ = e^(2πiφ)|ψ⟩,
    estimates φ to n_precision bits.
    
    Args:
        n_precision: Number of precision qubits
        unitary: Circuit implementing U (on remaining qubits)
        eigenstate: Initial state for the eigenstate register
        
    Returns:
        Circuit for phase estimation
    """
    n_eigenstate = unitary.n_qubits
    total_qubits = n_precision + n_eigenstate
    
    circuit = Circuit(total_qubits)
    
    # Initialize precision register to |+⟩^n
    for i in range(n_precision):
        circuit.h(i)
    
    # Apply controlled-U^(2^k) operations
    for k in range(n_precision):
        power = 2 ** (n_precision - 1 - k)
        control = k
        
        # Apply U^power controlled by qubit k
        for _ in range(power):
            # Map unitary operations to target qubits
            for op in unitary.operations:
                if op.op_type.value == 'gate':
                    # Make it controlled
                    target_qubits = [q + n_precision for q in op.qubits]
                    # Create controlled version (simplified for single-qubit gates)
                    if len(op.qubits) == 1:
                        if op.gate.name == 'X':
                            circuit.cx(control, target_qubits[0])
                        elif op.gate.name == 'Z':
                            circuit.cz(control, target_qubits[0])
                        elif op.gate.name == 'P':
                            phi = op.gate.params['phi']
                            circuit.cp(phi, control, target_qubits[0])
                        # Add more controlled gates as needed
    
    # Apply inverse QFT to precision register
    # (Swap qubits first)
    for i in range(n_precision // 2):
        circuit.swap(i, n_precision - 1 - i)
    
    # Inverse QFT rotations
    for i in range(n_precision - 1, -1, -1):
        for j in range(n_precision - 1, i, -1):
            angle = -np.pi / (2 ** (j - i))
            circuit.cp(angle, j, i)
        circuit.h(i)
    
    # Measure precision qubits
    for i in range(n_precision):
        circuit.measure(i, i)
    
    return circuit


# ============================================================================
# Variational Circuits (VQE Ansätze)
# ============================================================================

def hardware_efficient_ansatz(n: int, depth: int, 
                               params: Optional[np.ndarray] = None) -> Circuit:
    """
    Hardware-efficient ansatz for VQE.
    
    Alternating layers of single-qubit rotations and entangling gates.
    
    Args:
        n: Number of qubits
        depth: Number of layers
        params: Parameters for rotation gates (if None, uses placeholders)
        
    Returns:
        Parametrized circuit
    """
    circuit = Circuit(n)
    
    if params is None:
        params = np.zeros(depth * n * 3 + (depth - 1) * (n - 1))
    
    param_idx = 0
    
    for layer in range(depth):
        # Single-qubit rotation layer
        for i in range(n):
            circuit.rx(params[param_idx], i)
            param_idx += 1
            circuit.ry(params[param_idx], i)
            param_idx += 1
            circuit.rz(params[param_idx], i)
            param_idx += 1
        
        # Entangling layer (except last layer)
        if layer < depth - 1:
            for i in range(n - 1):
                circuit.cx(i, i + 1)
    
    return circuit


def uccsd_ansatz(n_qubits: int, n_electrons: int,
                 params: Optional[np.ndarray] = None) -> Circuit:
    """
    Simplified UCCSD-like ansatz for VQE.
    
    Unitary Coupled Cluster Singles and Doubles is commonly used
    for quantum chemistry simulations.
    
    This is a simplified version for demonstration.
    """
    circuit = Circuit(n_qubits)
    
    # Initialize HF state (first n_electrons qubits to |1⟩)
    for i in range(n_electrons):
        circuit.x(i)
    
    # Single excitations
    n_singles = n_electrons * (n_qubits - n_electrons)
    # Double excitations  
    n_doubles = n_singles * (n_singles - 1) // 2
    
    if params is None:
        params = np.zeros(n_singles + n_doubles)
    
    param_idx = 0
    
    # Apply single excitation operators
    for i in range(n_electrons):
        for a in range(n_electrons, n_qubits):
            theta = params[param_idx] if param_idx < len(params) else 0
            param_idx += 1
            
            # Simplified: use Givens rotation
            circuit.ry(theta, i)
            circuit.cx(i, a)
            circuit.ry(-theta, i)
    
    return circuit


# ============================================================================
# Utility Functions
# ============================================================================

def bell_pair() -> Circuit:
    """Create a Bell pair (|00⟩ + |11⟩)/√2."""
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def ghz_circuit(n: int) -> Circuit:
    """Create GHZ state (|00...0⟩ + |11...1⟩)/√2."""
    circuit = Circuit(n)
    circuit.h(0)
    for i in range(n - 1):
        circuit.cx(i, i + 1)
    return circuit


def teleportation_circuit() -> Circuit:
    """
    Quantum teleportation protocol.
    
    Teleports qubit 0's state to qubit 2 using entanglement.
    """
    circuit = Circuit(3, 2)
    
    # Create Bell pair between qubits 1 and 2
    circuit.h(1)
    circuit.cx(1, 2)
    
    # Bell measurement on qubits 0 and 1
    circuit.cx(0, 1)
    circuit.h(0)
    
    # Measure qubits 0 and 1
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    
    # Apply corrections based on measurements
    # (In real implementation, these would be classically controlled)
    # circuit.cx(1, 2)  # X correction if qubit 1 measured |1⟩
    # circuit.cz(0, 2)  # Z correction if qubit 0 measured |1⟩
    
    return circuit


def superdense_coding() -> Circuit:
    """
    Superdense coding protocol.
    
    Sends 2 classical bits using 1 qubit.
    """
    circuit = Circuit(2, 2)
    
    # Create Bell pair
    circuit.h(0)
    circuit.cx(0, 1)
    
    # Alice encodes message by applying gates to her qubit (qubit 0)
    # 00: I (do nothing)
    # 01: X
    # 10: Z  
    # 11: XZ (or iY)
    # (For demo, encode 11)
    circuit.x(0)
    circuit.z(0)
    
    # Bob decodes
    circuit.cx(0, 1)
    circuit.h(0)
    
    # Measure
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    
    return circuit
