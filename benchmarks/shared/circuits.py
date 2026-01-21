"""
Common Test Circuits for Benchmarking

Provides circuit builders for NoeticEidos, Qiskit, and Cirq to enable
direct fidelity comparisons.
"""

import numpy as np
import sys
import os

# Add parent directory to path for importing our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from circuit import Circuit
from state import QuantumState


# =============================================================================
# NoeticEidos Circuit Builders
# =============================================================================

def build_bell_circuit_ours() -> Circuit:
    """Build Bell state circuit: |Φ+⟩ = (|00⟩ + |11⟩)/√2"""
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def build_ghz_circuit_ours(n_qubits: int) -> Circuit:
    """Build GHZ state circuit: (|00...0⟩ + |11...1⟩)/√2"""
    qc = Circuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def build_qft_circuit_ours(n_qubits: int) -> Circuit:
    """
    Build Quantum Fourier Transform circuit.

    QFT transforms computational basis to Fourier basis.
    """
    qc = Circuit(n_qubits)

    for i in range(n_qubits):
        qc.h(i)
        for j in range(i + 1, n_qubits):
            # Controlled phase rotation
            angle = np.pi / (2 ** (j - i))
            qc.cp(angle, j, i)

    # Swap qubits for standard QFT ordering
    for i in range(n_qubits // 2):
        qc.swap(i, n_qubits - 1 - i)

    return qc


def build_random_clifford_circuit_ours(n_qubits: int, depth: int = 10, seed: int = 42) -> Circuit:
    """
    Build a random Clifford circuit for testing.

    Clifford gates: H, S, CNOT generate the Clifford group.
    Uses deterministic random number consumption for reproducibility.
    """
    rng = np.random.RandomState(seed)
    qc = Circuit(n_qubits)

    for _ in range(depth):
        gate_type = rng.choice(['H', 'S', 'CNOT'])

        if gate_type == 'H':
            q = rng.randint(n_qubits)
            qc.h(q)
        elif gate_type == 'S':
            q = rng.randint(n_qubits)
            qc.s(q)
        else:  # CNOT
            if n_qubits > 1:
                # Deterministic: always consume exactly 2 random numbers
                control = rng.randint(n_qubits)
                target = rng.randint(n_qubits - 1)
                if target >= control:
                    target += 1  # Skip control index
                qc.cx(control, target)

    return qc


def build_hardware_efficient_ansatz_ours(params: np.ndarray, n_qubits: int) -> Circuit:
    """
    Build hardware-efficient ansatz for VQE.

    Structure: Ry rotations on each qubit + CNOT ladder
    """
    qc = Circuit(n_qubits)

    # Ry rotation layer
    for i in range(min(len(params), n_qubits)):
        qc.ry(params[i], i)

    # CNOT ladder
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    return qc


# =============================================================================
# Qiskit Circuit Builders
# =============================================================================

def build_bell_circuit_qiskit():
    """Build Bell state circuit in Qiskit."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def build_ghz_circuit_qiskit(n_qubits: int):
    """Build GHZ state circuit in Qiskit."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def build_qft_circuit_qiskit(n_qubits: int):
    """Build QFT circuit in Qiskit."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.h(i)
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            qc.cp(angle, j, i)

    # Swap for standard ordering
    for i in range(n_qubits // 2):
        qc.swap(i, n_qubits - 1 - i)

    return qc


def build_random_clifford_circuit_qiskit(n_qubits: int, depth: int = 10, seed: int = 42):
    """Build random Clifford circuit in Qiskit."""
    from qiskit import QuantumCircuit

    rng = np.random.RandomState(seed)
    qc = QuantumCircuit(n_qubits)

    for _ in range(depth):
        gate_type = rng.choice(['H', 'S', 'CNOT'])

        if gate_type == 'H':
            q = rng.randint(n_qubits)
            qc.h(q)
        elif gate_type == 'S':
            q = rng.randint(n_qubits)
            qc.s(q)
        else:
            if n_qubits > 1:
                # Deterministic: always consume exactly 2 random numbers
                control = rng.randint(n_qubits)
                target = rng.randint(n_qubits - 1)
                if target >= control:
                    target += 1  # Skip control index
                qc.cx(control, target)

    return qc


# =============================================================================
# Cirq Circuit Builders
# =============================================================================

def build_bell_circuit_cirq():
    """Build Bell state circuit in Cirq."""
    import cirq

    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1])
    ])
    return circuit


def build_ghz_circuit_cirq(n_qubits: int):
    """Build GHZ state circuit in Cirq."""
    import cirq

    qubits = cirq.LineQubit.range(n_qubits)
    ops = [cirq.H(qubits[0])]
    for i in range(n_qubits - 1):
        ops.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return cirq.Circuit(ops)


def build_qft_circuit_cirq(n_qubits: int):
    """Build QFT circuit in Cirq."""
    import cirq

    qubits = cirq.LineQubit.range(n_qubits)
    ops = []

    for i in range(n_qubits):
        ops.append(cirq.H(qubits[i]))
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            ops.append(cirq.CZPowGate(exponent=angle / np.pi)(qubits[j], qubits[i]))

    # Swap for standard ordering
    for i in range(n_qubits // 2):
        ops.append(cirq.SWAP(qubits[i], qubits[n_qubits - 1 - i]))

    return cirq.Circuit(ops)


def build_random_clifford_circuit_cirq(n_qubits: int, depth: int = 10, seed: int = 42):
    """Build random Clifford circuit in Cirq."""
    import cirq

    rng = np.random.RandomState(seed)
    qubits = cirq.LineQubit.range(n_qubits)
    ops = []

    for _ in range(depth):
        gate_type = rng.choice(['H', 'S', 'CNOT'])

        if gate_type == 'H':
            q = rng.randint(n_qubits)
            ops.append(cirq.H(qubits[q]))
        elif gate_type == 'S':
            q = rng.randint(n_qubits)
            ops.append(cirq.S(qubits[q]))
        else:
            if n_qubits > 1:
                # Deterministic: always consume exactly 2 random numbers
                control = rng.randint(n_qubits)
                target = rng.randint(n_qubits - 1)
                if target >= control:
                    target += 1  # Skip control index
                ops.append(cirq.CNOT(qubits[control], qubits[target]))

    return cirq.Circuit(ops)


# =============================================================================
# State Vector Extraction
# =============================================================================

def get_statevector_ours(circuit: Circuit) -> np.ndarray:
    """Get statevector from our circuit."""
    return circuit.statevector()


def get_statevector_qiskit(circuit) -> np.ndarray:
    """Get statevector from Qiskit circuit.

    Note: Qiskit uses little-endian qubit ordering (qubit 0 = rightmost bit),
    while our implementation uses big-endian (qubit 0 = leftmost bit).
    We reverse the qubit order to match our convention.
    """
    from qiskit_aer import AerSimulator

    circuit_with_save = circuit.copy()
    circuit_with_save.save_statevector()

    simulator = AerSimulator(method='statevector')
    result = simulator.run(circuit_with_save).result()
    sv = np.array(result.get_statevector())

    # Convert from Qiskit's little-endian to our big-endian convention
    n_qubits = int(np.log2(len(sv)))
    sv_reshaped = sv.reshape([2] * n_qubits)
    sv_reordered = sv_reshaped.transpose(list(reversed(range(n_qubits))))
    return sv_reordered.flatten()


def get_statevector_cirq(circuit) -> np.ndarray:
    """Get statevector from Cirq circuit."""
    import cirq

    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    return result.final_state_vector
