"""
Quantum Gates

Unitary transformations on quantum states. Each gate preserves the 
unit norm of state vectors and corresponds to rotations on the 
Bloch sphere (for single qubits) or more general SO(2^n) rotations.

Geometric Interpretation:
- Single-qubit gates: SU(2) rotations on Bloch sphere
- Pauli gates: π rotations around x, y, z axes
- Hadamard: rotation to/from diagonal axis
- Phase gates: rotations around z-axis
- Multi-qubit gates: Entangling operations in larger Hilbert space
"""

import numpy as np
from typing import List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import functools


# ============================================================================
# Gate Definitions
# ============================================================================

# Pauli Matrices
PAULI_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# Hadamard
HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

# Phase gates
S_GATE = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
S_DAG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
T_DAG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)

# CNOT (Controlled-X)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=np.complex128)

# CZ (Controlled-Z)
CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=np.complex128)

# SWAP
SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=np.complex128)

# Toffoli (CCNOT)
TOFFOLI = np.eye(8, dtype=np.complex128)
TOFFOLI[6, 6] = 0
TOFFOLI[7, 7] = 0
TOFFOLI[6, 7] = 1
TOFFOLI[7, 6] = 1


class GateType(Enum):
    """Classification of quantum gates."""
    SINGLE = "single"
    TWO_QUBIT = "two_qubit"
    MULTI_QUBIT = "multi_qubit"
    PARAMETRIC = "parametric"


@dataclass
class Gate:
    """
    Quantum gate with metadata for circuit manipulation.
    
    Attributes:
        name: Human-readable gate name
        matrix: Unitary matrix representation
        n_qubits: Number of qubits the gate acts on
        gate_type: Classification
        params: Parameters for parametric gates
    """
    name: str
    matrix: np.ndarray
    n_qubits: int
    gate_type: GateType
    params: Optional[dict] = None
    
    def __post_init__(self):
        # Verify unitarity
        dim = 2 ** self.n_qubits
        if self.matrix.shape != (dim, dim):
            raise ValueError(f"Gate matrix shape {self.matrix.shape} doesn't match {self.n_qubits} qubits")
        
        identity = self.matrix @ self.matrix.conj().T
        if not np.allclose(identity, np.eye(dim)):
            raise ValueError(f"Gate {self.name} is not unitary")
    
    @property
    def adjoint(self) -> 'Gate':
        """Return the adjoint (conjugate transpose) of this gate."""
        return Gate(
            name=f"{self.name}†",
            matrix=self.matrix.conj().T,
            n_qubits=self.n_qubits,
            gate_type=self.gate_type,
            params=self.params
        )
    
    def __repr__(self) -> str:
        return f"Gate({self.name}, {self.n_qubits} qubit(s))"


# ============================================================================
# Standard Gate Constructors
# ============================================================================

def I() -> Gate:
    """Identity gate."""
    return Gate("I", PAULI_I.copy(), 1, GateType.SINGLE)


def X() -> Gate:
    """Pauli-X (NOT) gate: π rotation around x-axis."""
    return Gate("X", PAULI_X.copy(), 1, GateType.SINGLE)


def Y() -> Gate:
    """Pauli-Y gate: π rotation around y-axis."""
    return Gate("Y", PAULI_Y.copy(), 1, GateType.SINGLE)


def Z() -> Gate:
    """Pauli-Z gate: π rotation around z-axis."""
    return Gate("Z", PAULI_Z.copy(), 1, GateType.SINGLE)


def H() -> Gate:
    """Hadamard gate: rotation to diagonal basis."""
    return Gate("H", HADAMARD.copy(), 1, GateType.SINGLE)


def S() -> Gate:
    """S gate (√Z): π/2 rotation around z-axis."""
    return Gate("S", S_GATE.copy(), 1, GateType.SINGLE)


def Sdg() -> Gate:
    """S-dagger gate: -π/2 rotation around z-axis."""
    return Gate("S†", S_DAG.copy(), 1, GateType.SINGLE)


def T() -> Gate:
    """T gate (√S): π/4 rotation around z-axis."""
    return Gate("T", T_GATE.copy(), 1, GateType.SINGLE)


def Tdg() -> Gate:
    """T-dagger gate: -π/4 rotation around z-axis."""
    return Gate("T†", T_DAG.copy(), 1, GateType.SINGLE)


# ============================================================================
# Parametric Single-Qubit Gates
# ============================================================================

def Rx(theta: float) -> Gate:
    """
    X-rotation gate: exp(-i θ/2 X)
    
    Rotates around x-axis by angle theta.
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    matrix = np.array([
        [c, -1j * s],
        [-1j * s, c]
    ], dtype=np.complex128)
    return Gate("Rx", matrix, 1, GateType.PARAMETRIC, params={'theta': theta})


def Ry(theta: float) -> Gate:
    """
    Y-rotation gate: exp(-i θ/2 Y)
    
    Rotates around y-axis by angle theta.
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    matrix = np.array([
        [c, -s],
        [s, c]
    ], dtype=np.complex128)
    return Gate("Ry", matrix, 1, GateType.PARAMETRIC, params={'theta': theta})


def Rz(theta: float) -> Gate:
    """
    Z-rotation gate: exp(-i θ/2 Z)
    
    Rotates around z-axis by angle theta.
    """
    matrix = np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=np.complex128)
    return Gate("Rz", matrix, 1, GateType.PARAMETRIC, params={'theta': theta})


def P(phi: float) -> Gate:
    """
    Phase gate: diag(1, e^(iφ))
    
    Also known as R1 or Rφ gate.
    """
    matrix = np.array([
        [1, 0],
        [0, np.exp(1j * phi)]
    ], dtype=np.complex128)
    return Gate("P", matrix, 1, GateType.PARAMETRIC, params={'phi': phi})


def U3(theta: float, phi: float, lam: float) -> Gate:
    """
    General single-qubit gate with 3 Euler angles.
    
    U3(θ, φ, λ) = Rz(φ) Ry(θ) Rz(λ)
    
    Any single-qubit unitary can be decomposed into this form.
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    matrix = np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=np.complex128)
    return Gate("U3", matrix, 1, GateType.PARAMETRIC, 
                params={'theta': theta, 'phi': phi, 'lambda': lam})


# ============================================================================
# Two-Qubit Gates
# ============================================================================

def CX() -> Gate:
    """CNOT (Controlled-X) gate."""
    return Gate("CX", CNOT.copy(), 2, GateType.TWO_QUBIT)


def CY() -> Gate:
    """Controlled-Y gate."""
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0]
    ], dtype=np.complex128)
    return Gate("CY", matrix, 2, GateType.TWO_QUBIT)


def CZ_gate() -> Gate:
    """Controlled-Z gate."""
    return Gate("CZ", CZ.copy(), 2, GateType.TWO_QUBIT)


def SWAP_gate() -> Gate:
    """SWAP gate: exchanges two qubits."""
    return Gate("SWAP", SWAP.copy(), 2, GateType.TWO_QUBIT)


def iSWAP() -> Gate:
    """iSWAP gate: SWAP with phase."""
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.complex128)
    return Gate("iSWAP", matrix, 2, GateType.TWO_QUBIT)


def SQRTSWAP() -> Gate:
    """Square root of SWAP gate."""
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 0.5*(1+1j), 0.5*(1-1j), 0],
        [0, 0.5*(1-1j), 0.5*(1+1j), 0],
        [0, 0, 0, 1]
    ], dtype=np.complex128)
    return Gate("√SWAP", matrix, 2, GateType.TWO_QUBIT)


def CRx(theta: float) -> Gate:
    """Controlled-Rx gate."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, c, -1j * s],
        [0, 0, -1j * s, c]
    ], dtype=np.complex128)
    return Gate("CRx", matrix, 2, GateType.PARAMETRIC, params={'theta': theta})


def CRy(theta: float) -> Gate:
    """Controlled-Ry gate."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, c, -s],
        [0, 0, s, c]
    ], dtype=np.complex128)
    return Gate("CRy", matrix, 2, GateType.PARAMETRIC, params={'theta': theta})


def CRz(theta: float) -> Gate:
    """Controlled-Rz gate."""
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.exp(-1j * theta / 2), 0],
        [0, 0, 0, np.exp(1j * theta / 2)]
    ], dtype=np.complex128)
    return Gate("CRz", matrix, 2, GateType.PARAMETRIC, params={'theta': theta})


def CP(phi: float) -> Gate:
    """Controlled-Phase gate."""
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * phi)]
    ], dtype=np.complex128)
    return Gate("CP", matrix, 2, GateType.PARAMETRIC, params={'phi': phi})


def XX(theta: float) -> Gate:
    """XX interaction gate: exp(-i θ/2 X⊗X)"""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    matrix = np.array([
        [c, 0, 0, -1j * s],
        [0, c, -1j * s, 0],
        [0, -1j * s, c, 0],
        [-1j * s, 0, 0, c]
    ], dtype=np.complex128)
    return Gate("XX", matrix, 2, GateType.PARAMETRIC, params={'theta': theta})


def YY(theta: float) -> Gate:
    """YY interaction gate: exp(-i θ/2 Y⊗Y)"""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    matrix = np.array([
        [c, 0, 0, 1j * s],
        [0, c, -1j * s, 0],
        [0, -1j * s, c, 0],
        [1j * s, 0, 0, c]
    ], dtype=np.complex128)
    return Gate("YY", matrix, 2, GateType.PARAMETRIC, params={'theta': theta})


def ZZ(theta: float) -> Gate:
    """ZZ interaction gate: exp(-i θ/2 Z⊗Z)"""
    matrix = np.diag([
        np.exp(-1j * theta / 2),
        np.exp(1j * theta / 2),
        np.exp(1j * theta / 2),
        np.exp(-1j * theta / 2)
    ])
    return Gate("ZZ", matrix, 2, GateType.PARAMETRIC, params={'theta': theta})


# ============================================================================
# Multi-Qubit Gates
# ============================================================================

def CCX() -> Gate:
    """Toffoli (CCNOT) gate: controlled-controlled-X."""
    return Gate("CCX", TOFFOLI.copy(), 3, GateType.MULTI_QUBIT)


def CCZ() -> Gate:
    """Controlled-controlled-Z gate."""
    matrix = np.eye(8, dtype=np.complex128)
    matrix[7, 7] = -1
    return Gate("CCZ", matrix, 3, GateType.MULTI_QUBIT)


def CSWAP() -> Gate:
    """Fredkin (controlled-SWAP) gate."""
    matrix = np.eye(8, dtype=np.complex128)
    # Swap |110⟩ and |101⟩
    matrix[5, 5] = 0
    matrix[6, 6] = 0
    matrix[5, 6] = 1
    matrix[6, 5] = 1
    return Gate("CSWAP", matrix, 3, GateType.MULTI_QUBIT)


# ============================================================================
# Gate Application Utilities
# ============================================================================

def apply_gate(gate: Gate, amplitudes: np.ndarray, 
               target_qubits: List[int], n_qubits: int) -> np.ndarray:
    """
    Apply a gate to specific qubits in a state vector.
    
    Args:
        gate: The gate to apply
        amplitudes: Current state vector
        target_qubits: Which qubits the gate acts on
        n_qubits: Total number of qubits in the system
        
    Returns:
        New state vector after gate application
    """
    if len(target_qubits) != gate.n_qubits:
        raise ValueError(f"Gate {gate.name} needs {gate.n_qubits} qubits, got {len(target_qubits)}")
    
    # Build the full operator via tensor products
    full_operator = _expand_gate(gate.matrix, target_qubits, n_qubits)
    return full_operator @ amplitudes


def _expand_gate(gate_matrix: np.ndarray, target_qubits: List[int], 
                 n_qubits: int) -> np.ndarray:
    """
    Expand a gate matrix to act on the full Hilbert space.
    
    Uses the approach of constructing the permutation that moves
    target qubits to the front, applying the gate, then permuting back.
    """
    dim = 2 ** n_qubits
    gate_dim = gate_matrix.shape[0]
    n_gate_qubits = len(target_qubits)
    
    # Create the full unitary by computing action on each basis state
    full_matrix = np.zeros((dim, dim), dtype=np.complex128)
    
    for i in range(dim):
        # Extract bits for target qubits
        target_bits = 0
        for idx, q in enumerate(target_qubits):
            bit = (i >> (n_qubits - 1 - q)) & 1
            target_bits |= (bit << (n_gate_qubits - 1 - idx))
        
        # Apply gate to target bits
        for j in range(gate_dim):
            if np.abs(gate_matrix[j, target_bits]) < 1e-15:
                continue
            
            # Construct output basis state
            output_idx = i
            for idx, q in enumerate(target_qubits):
                # Clear the old bit
                output_idx &= ~(1 << (n_qubits - 1 - q))
                # Set the new bit
                new_bit = (j >> (n_gate_qubits - 1 - idx)) & 1
                output_idx |= (new_bit << (n_qubits - 1 - q))
            
            full_matrix[output_idx, i] += gate_matrix[j, target_bits]
    
    return full_matrix


def tensor_gates(*gates: Gate) -> Gate:
    """Compute tensor product of gates: G1 ⊗ G2 ⊗ ..."""
    matrix = gates[0].matrix
    total_qubits = gates[0].n_qubits
    name_parts = [gates[0].name]
    
    for g in gates[1:]:
        matrix = np.kron(matrix, g.matrix)
        total_qubits += g.n_qubits
        name_parts.append(g.name)
    
    return Gate(
        name="⊗".join(name_parts),
        matrix=matrix,
        n_qubits=total_qubits,
        gate_type=GateType.MULTI_QUBIT if total_qubits > 2 else GateType.TWO_QUBIT
    )


def controlled(gate: Gate, n_controls: int = 1) -> Gate:
    """
    Create a controlled version of a gate.
    
    Args:
        gate: Base gate to control
        n_controls: Number of control qubits
        
    Returns:
        Controlled gate (C^n-U)
    """
    base_dim = gate.matrix.shape[0]
    new_dim = base_dim * (2 ** n_controls)
    n_qubits = gate.n_qubits + n_controls
    
    matrix = np.eye(new_dim, dtype=np.complex128)
    
    # The gate acts only when all control qubits are |1⟩
    # That's the last `base_dim` entries
    control_on_start = new_dim - base_dim
    matrix[control_on_start:, control_on_start:] = gate.matrix
    
    name = "C" * n_controls + gate.name
    return Gate(name, matrix, n_qubits, GateType.MULTI_QUBIT)


# ============================================================================
# Gate Decomposition Utilities  
# ============================================================================

def decompose_to_u3(gate: Gate) -> Optional[dict]:
    """
    Decompose a single-qubit gate into U3 parameters.
    
    Any U ∈ SU(2) can be written as e^(iα) U3(θ, φ, λ).
    
    Returns dict with 'theta', 'phi', 'lambda', 'phase' or None if not single-qubit.
    """
    if gate.n_qubits != 1:
        return None
    
    U = gate.matrix
    
    # Extract global phase to get SU(2) element
    det = np.linalg.det(U)
    phase = np.angle(det) / 2
    V = U * np.exp(-1j * phase)  # Now V ∈ SU(2)
    
    # Extract Euler angles
    # V = [[cos(θ/2), -e^(iλ)sin(θ/2)], [e^(iφ)sin(θ/2), e^(i(φ+λ))cos(θ/2)]]
    
    theta = 2 * np.arccos(np.clip(np.abs(V[0, 0]), 0, 1))
    
    if np.abs(np.sin(theta/2)) < 1e-10:
        # θ ≈ 0, gate is essentially a phase
        phi = 0
        lam = np.angle(V[1, 1])
    elif np.abs(np.cos(theta/2)) < 1e-10:
        # θ ≈ π
        phi = np.angle(V[1, 0])
        lam = -np.angle(V[0, 1])
    else:
        phi = np.angle(V[1, 0])
        lam = np.angle(-V[0, 1])
    
    return {
        'theta': theta,
        'phi': phi,
        'lambda': lam,
        'global_phase': phase
    }


# ============================================================================
# Gate Library Registry
# ============================================================================

GATE_LIBRARY = {
    # Single-qubit
    'I': I, 'X': X, 'Y': Y, 'Z': Z,
    'H': H, 'S': S, 'SDG': Sdg, 'T': T, 'TDG': Tdg,
    'RX': Rx, 'RY': Ry, 'RZ': Rz, 'P': P, 'U3': U3,
    # Two-qubit
    'CX': CX, 'CNOT': CX, 'CY': CY, 'CZ': CZ_gate,
    'SWAP': SWAP_gate, 'ISWAP': iSWAP, 'SQRTSWAP': SQRTSWAP,
    'CRX': CRx, 'CRY': CRy, 'CRZ': CRz, 'CP': CP,
    'XX': XX, 'YY': YY, 'ZZ': ZZ,
    # Multi-qubit
    'CCX': CCX, 'TOFFOLI': CCX, 'CCZ': CCZ, 'CSWAP': CSWAP, 'FREDKIN': CSWAP,
}


def get_gate(name: str, *params) -> Gate:
    """Get a gate from the library by name."""
    name_upper = name.upper()
    if name_upper not in GATE_LIBRARY:
        raise ValueError(f"Unknown gate: {name}")
    
    factory = GATE_LIBRARY[name_upper]
    if params:
        return factory(*params)
    return factory()
