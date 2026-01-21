"""
Test Hamiltonians for Benchmarking

Standard Hamiltonians for VQE, noise modeling, and QFI tracking.
"""

import numpy as np


# =============================================================================
# Pauli Matrices
# =============================================================================

I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def pauli_z(n_qubits: int = 1, qubit: int = 0) -> np.ndarray:
    """
    Single-qubit Pauli-Z operator on specified qubit.

    Used as generator for QFI tracking.
    """
    result = np.array([[1]], dtype=np.complex128)

    for i in range(n_qubits):
        if i == qubit:
            result = np.kron(result, Z)
        else:
            result = np.kron(result, I)

    return result


def pauli_x(n_qubits: int = 1, qubit: int = 0) -> np.ndarray:
    """Single-qubit Pauli-X operator on specified qubit."""
    result = np.array([[1]], dtype=np.complex128)

    for i in range(n_qubits):
        if i == qubit:
            result = np.kron(result, X)
        else:
            result = np.kron(result, I)

    return result


# =============================================================================
# VQE Test Hamiltonians
# =============================================================================

def zz_hamiltonian(n_qubits: int = 2) -> np.ndarray:
    """
    Z⊗Z Hamiltonian for 2-qubit VQE.

    H = Z ⊗ Z

    Ground state: |00⟩ or |11⟩ with energy -1
    Excited state: |01⟩ or |10⟩ with energy +1

    This is a simple benchmark Hamiltonian where:
    - Ground state energy = -1
    - Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 achieves E = -1
    """
    if n_qubits != 2:
        raise ValueError("ZZ Hamiltonian is defined for 2 qubits")

    return np.kron(Z, Z)


def transverse_field_ising(n_qubits: int, J: float = 1.0, h: float = 0.5) -> np.ndarray:
    """
    Transverse field Ising model Hamiltonian.

    H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ

    Args:
        n_qubits: Number of qubits (spins)
        J: Coupling strength
        h: Transverse field strength
    """
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)

    # ZZ interactions
    for i in range(n_qubits - 1):
        ZZ = np.array([[1]], dtype=np.complex128)
        for j in range(n_qubits):
            if j == i or j == i + 1:
                ZZ = np.kron(ZZ, Z)
            else:
                ZZ = np.kron(ZZ, I)
        H -= J * ZZ

    # Transverse field
    for i in range(n_qubits):
        H -= h * pauli_x(n_qubits, i)

    return H


def heisenberg_hamiltonian(n_qubits: int, J: float = 1.0) -> np.ndarray:
    """
    Heisenberg XXX model Hamiltonian.

    H = J Σᵢ (XᵢXᵢ₊₁ + YᵢYᵢ₊₁ + ZᵢZᵢ₊₁)

    Args:
        n_qubits: Number of qubits
        J: Coupling strength
    """
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(n_qubits - 1):
        for pauli in [X, Y, Z]:
            term = np.array([[1]], dtype=np.complex128)
            for j in range(n_qubits):
                if j == i or j == i + 1:
                    term = np.kron(term, pauli)
                else:
                    term = np.kron(term, I)
            H += J * term

    return H


# =============================================================================
# Single-Qubit Hamiltonians (for noise geometry benchmarks)
# =============================================================================

def single_qubit_hamiltonian(omega: float = 1.0) -> np.ndarray:
    """
    Simple single-qubit Hamiltonian: H = (ω/2) σz

    Generates rotation about Z-axis at frequency ω.
    """
    return (omega / 2) * Z


def rabi_hamiltonian(omega: float = 1.0, delta: float = 0.5) -> np.ndarray:
    """
    Rabi model Hamiltonian for driven two-level system.

    H = (δ/2) σz + (Ω/2) σx

    where δ is detuning and Ω is Rabi frequency.
    """
    return (delta / 2) * Z + (omega / 2) * X


# =============================================================================
# Utility Functions
# =============================================================================

def ground_state_energy(H: np.ndarray) -> float:
    """Compute exact ground state energy of Hamiltonian."""
    eigenvalues = np.linalg.eigvalsh(H)
    return float(np.min(eigenvalues))


def ground_state(H: np.ndarray) -> np.ndarray:
    """Compute ground state eigenvector of Hamiltonian."""
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    idx = np.argmin(eigenvalues)
    return eigenvectors[:, idx]


def spectral_gap(H: np.ndarray) -> float:
    """Compute spectral gap (E₁ - E₀) of Hamiltonian."""
    eigenvalues = np.sort(np.linalg.eigvalsh(H))
    if len(eigenvalues) < 2:
        return 0.0
    return float(eigenvalues[1] - eigenvalues[0])
