"""
Quantum State Representation

Core state vector and density matrix implementations with proper
complex amplitude handling. Bridges to information geometry through
the Fubini-Study metric on projective Hilbert space.

Mathematical Foundation:
- Pure states: |ψ⟩ ∈ ℂⁿ with ⟨ψ|ψ⟩ = 1
- Mixed states: ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ| with Tr(ρ) = 1
- Measurement: P(outcome=k) = ⟨ψ|Πₖ|ψ⟩ = Tr(ρ Πₖ)
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class StateType(Enum):
    """Classification of quantum states."""
    PURE = "pure"
    MIXED = "mixed"


@dataclass
class QuantumState:
    """
    Quantum state representation supporting both pure and mixed states.
    
    For n qubits, the state lives in ℂ^(2^n).
    
    Attributes:
        n_qubits: Number of qubits
        amplitudes: Complex state vector (for pure states)
        density_matrix: Density operator (for mixed states or computed from pure)
        state_type: Whether state is pure or mixed
    """
    n_qubits: int
    amplitudes: Optional[np.ndarray] = None
    density_matrix: Optional[np.ndarray] = None
    state_type: StateType = StateType.PURE
    
    def __post_init__(self):
        self.dim = 2 ** self.n_qubits
        
        if self.amplitudes is None and self.density_matrix is None:
            # Default to |00...0⟩ state
            self.amplitudes = np.zeros(self.dim, dtype=np.complex128)
            self.amplitudes[0] = 1.0
        
        if self.amplitudes is not None:
            self.amplitudes = np.asarray(self.amplitudes, dtype=np.complex128)
            self._normalize()
            
        if self.density_matrix is not None:
            self.density_matrix = np.asarray(self.density_matrix, dtype=np.complex128)
            self.state_type = StateType.MIXED
    
    def _normalize(self):
        """Ensure unit norm for state vector."""
        if self.amplitudes is not None:
            norm = np.linalg.norm(self.amplitudes)
            if norm > 1e-10:
                self.amplitudes /= norm
    
    @property
    def rho(self) -> np.ndarray:
        """
        Get density matrix representation.
        
        For pure state |ψ⟩: ρ = |ψ⟩⟨ψ|
        For mixed state: return stored density matrix
        """
        if self.state_type == StateType.PURE and self.amplitudes is not None:
            return np.outer(self.amplitudes, np.conj(self.amplitudes))
        return self.density_matrix
    
    @property
    def purity(self) -> float:
        """
        Compute Tr(ρ²) - the purity of the state.
        
        Returns 1 for pure states, < 1 for mixed states.
        Related to linear entropy: S_L = 1 - Tr(ρ²)
        """
        rho = self.rho
        return np.real(np.trace(rho @ rho))
    
    @property
    def von_neumann_entropy(self) -> float:
        """
        Compute von Neumann entropy: S = -Tr(ρ log ρ)
        
        This is the quantum analog of Shannon entropy and measures
        the mixedness of the quantum state.
        """
        rho = self.rho
        eigenvalues = np.linalg.eigvalsh(rho)
        # Filter small/negative eigenvalues (numerical noise)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    def probability(self, basis_state: int) -> float:
        """
        Probability of measuring a specific computational basis state.
        
        Args:
            basis_state: Integer representation of basis state (e.g., 0 for |00⟩)
            
        Returns:
            P(outcome = basis_state)
        """
        if self.state_type == StateType.PURE:
            return np.abs(self.amplitudes[basis_state]) ** 2
        return np.real(self.rho[basis_state, basis_state])
    
    def probabilities(self) -> np.ndarray:
        """Get probability distribution over computational basis."""
        if self.state_type == StateType.PURE:
            return np.abs(self.amplitudes) ** 2
        return np.real(np.diag(self.rho))
    
    def measure(self, collapse: bool = True) -> Tuple[int, 'QuantumState']:
        """
        Perform measurement in computational basis.
        
        Args:
            collapse: Whether to collapse the state after measurement
            
        Returns:
            (outcome, post_measurement_state)
        """
        probs = self.probabilities()
        outcome = np.random.choice(self.dim, p=probs)
        
        if collapse:
            # Collapse to measured basis state
            new_amplitudes = np.zeros(self.dim, dtype=np.complex128)
            new_amplitudes[outcome] = 1.0
            return outcome, QuantumState(self.n_qubits, amplitudes=new_amplitudes)
        
        return outcome, self
    
    def measure_qubit(self, qubit_index: int, collapse: bool = True) -> Tuple[int, 'QuantumState']:
        """
        Measure a single qubit.
        
        Args:
            qubit_index: Which qubit to measure (0-indexed from left)
            collapse: Whether to collapse the state
            
        Returns:
            (outcome ∈ {0,1}, post_measurement_state)
        """
        # Calculate probability of measuring |0⟩ on this qubit
        p0 = 0.0
        for i in range(self.dim):
            # Check if qubit_index bit is 0 in basis state i
            if not (i >> (self.n_qubits - 1 - qubit_index)) & 1:
                p0 += self.probability(i)
        
        outcome = 0 if np.random.random() < p0 else 1
        
        if collapse:
            # Project onto subspace where this qubit has measured value
            new_amplitudes = np.zeros(self.dim, dtype=np.complex128)
            norm = 0.0
            
            for i in range(self.dim):
                bit_val = (i >> (self.n_qubits - 1 - qubit_index)) & 1
                if bit_val == outcome:
                    new_amplitudes[i] = self.amplitudes[i]
                    norm += np.abs(self.amplitudes[i]) ** 2
            
            new_amplitudes /= np.sqrt(norm)
            return outcome, QuantumState(self.n_qubits, amplitudes=new_amplitudes)
        
        return outcome, self
    
    def expectation(self, observable: np.ndarray) -> complex:
        """
        Compute expectation value ⟨O⟩ = Tr(ρO).
        
        Args:
            observable: Hermitian operator as matrix
            
        Returns:
            Expected value of the observable
        """
        return np.trace(self.rho @ observable)
    
    def fidelity(self, other: 'QuantumState') -> float:
        """
        Compute fidelity F(ρ, σ) between this state and another.
        
        For pure states: F = |⟨ψ|φ⟩|²
        For mixed states: F = (Tr√(√ρ σ √ρ))²
        """
        if self.state_type == StateType.PURE and other.state_type == StateType.PURE:
            overlap = np.abs(np.vdot(self.amplitudes, other.amplitudes))
            return overlap ** 2
        
        # General case using density matrices
        rho = self.rho
        sigma = other.rho
        sqrt_rho = _matrix_sqrt(rho)
        inner = sqrt_rho @ sigma @ sqrt_rho
        sqrt_inner = _matrix_sqrt(inner)
        return np.real(np.trace(sqrt_inner)) ** 2
    
    def bloch_vector(self) -> Optional[np.ndarray]:
        """
        Get Bloch sphere coordinates for single-qubit state.
        
        Returns [x, y, z] where:
        - x = Tr(ρσₓ)
        - y = Tr(ρσᵧ)  
        - z = Tr(ρσᵤ)
        
        For pure states, this lies on the sphere surface (|r| = 1).
        For mixed states, |r| < 1.
        """
        if self.n_qubits != 1:
            return None
        
        rho = self.rho
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        
        x = np.real(np.trace(rho @ sigma_x))
        y = np.real(np.trace(rho @ sigma_y))
        z = np.real(np.trace(rho @ sigma_z))
        
        return np.array([x, y, z])
    
    def partial_trace(self, keep_qubits: List[int]) -> 'QuantumState':
        """
        Compute partial trace, tracing out all qubits not in keep_qubits.
        
        This gives the reduced density matrix for a subsystem.
        
        Args:
            keep_qubits: Indices of qubits to keep
            
        Returns:
            Reduced state on the kept qubits
        """
        trace_out = [i for i in range(self.n_qubits) if i not in keep_qubits]
        
        if not trace_out:
            return self
        
        rho = self.rho
        n_keep = len(keep_qubits)
        dim_keep = 2 ** n_keep
        
        # Reorder qubits: kept first, traced second
        # Then trace out the second subsystem
        reduced_rho = np.zeros((dim_keep, dim_keep), dtype=np.complex128)
        
        for i in range(dim_keep):
            for j in range(dim_keep):
                for k in range(2 ** len(trace_out)):
                    # Construct full indices
                    idx_i = self._expand_index(i, keep_qubits, k, trace_out)
                    idx_j = self._expand_index(j, keep_qubits, k, trace_out)
                    reduced_rho[i, j] += rho[idx_i, idx_j]
        
        return QuantumState(n_keep, density_matrix=reduced_rho)
    
    def _expand_index(self, keep_val: int, keep_qubits: List[int], 
                      trace_val: int, trace_qubits: List[int]) -> int:
        """Helper to construct full basis index from partial indices."""
        result = 0
        keep_pos = 0
        trace_pos = 0
        
        for q in range(self.n_qubits):
            bit_pos = self.n_qubits - 1 - q
            if q in keep_qubits:
                bit = (keep_val >> (len(keep_qubits) - 1 - keep_pos)) & 1
                keep_pos += 1
            else:
                bit = (trace_val >> (len(trace_qubits) - 1 - trace_pos)) & 1
                trace_pos += 1
            result |= (bit << bit_pos)
        
        return result
    
    def tensor(self, other: 'QuantumState') -> 'QuantumState':
        """
        Compute tensor product |ψ⟩ ⊗ |φ⟩.
        
        This is how we compose multi-qubit systems.
        """
        if self.state_type == StateType.PURE and other.state_type == StateType.PURE:
            new_amplitudes = np.kron(self.amplitudes, other.amplitudes)
            return QuantumState(self.n_qubits + other.n_qubits, amplitudes=new_amplitudes)
        
        new_rho = np.kron(self.rho, other.rho)
        return QuantumState(self.n_qubits + other.n_qubits, density_matrix=new_rho)
    
    def copy(self) -> 'QuantumState':
        """Create a copy of this state."""
        if self.state_type == StateType.PURE:
            return QuantumState(self.n_qubits, amplitudes=self.amplitudes.copy())
        return QuantumState(self.n_qubits, density_matrix=self.density_matrix.copy())
    
    def __repr__(self) -> str:
        return f"QuantumState(n_qubits={self.n_qubits}, type={self.state_type.value}, purity={self.purity:.4f})"
    
    def to_ket_notation(self, threshold: float = 1e-6) -> str:
        """Pretty print state in Dirac notation."""
        if self.state_type != StateType.PURE:
            return f"Mixed state with purity {self.purity:.4f}"
        
        terms = []
        for i, amp in enumerate(self.amplitudes):
            if np.abs(amp) > threshold:
                basis = format(i, f'0{self.n_qubits}b')
                if np.abs(amp.imag) < threshold:
                    coef = f"{amp.real:+.3f}"
                elif np.abs(amp.real) < threshold:
                    coef = f"{amp.imag:+.3f}i"
                else:
                    coef = f"({amp.real:.3f}{amp.imag:+.3f}i)"
                terms.append(f"{coef}|{basis}⟩")
        
        return " ".join(terms) if terms else "|0⟩"


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Compute matrix square root via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T


# ============================================================================
# Standard State Constructors
# ============================================================================

def computational_basis(n_qubits: int, index: int = 0) -> QuantumState:
    """Create computational basis state |index⟩."""
    amplitudes = np.zeros(2 ** n_qubits, dtype=np.complex128)
    amplitudes[index] = 1.0
    return QuantumState(n_qubits, amplitudes=amplitudes)


def plus_state(n_qubits: int = 1) -> QuantumState:
    """Create |+⟩^⊗n state (uniform superposition)."""
    dim = 2 ** n_qubits
    amplitudes = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    return QuantumState(n_qubits, amplitudes=amplitudes)


def minus_state() -> QuantumState:
    """Create |−⟩ = (|0⟩ - |1⟩)/√2."""
    amplitudes = np.array([1, -1], dtype=np.complex128) / np.sqrt(2)
    return QuantumState(1, amplitudes=amplitudes)


def bell_state(which: int = 0) -> QuantumState:
    """
    Create Bell state.
    
    0: |Φ+⟩ = (|00⟩ + |11⟩)/√2
    1: |Φ−⟩ = (|00⟩ - |11⟩)/√2
    2: |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    3: |Ψ−⟩ = (|01⟩ - |10⟩)/√2
    """
    amplitudes = np.zeros(4, dtype=np.complex128)
    
    if which == 0:
        amplitudes[0] = amplitudes[3] = 1/np.sqrt(2)
    elif which == 1:
        amplitudes[0] = 1/np.sqrt(2)
        amplitudes[3] = -1/np.sqrt(2)
    elif which == 2:
        amplitudes[1] = amplitudes[2] = 1/np.sqrt(2)
    elif which == 3:
        amplitudes[1] = 1/np.sqrt(2)
        amplitudes[2] = -1/np.sqrt(2)
    
    return QuantumState(2, amplitudes=amplitudes)


def ghz_state(n_qubits: int) -> QuantumState:
    """
    Create GHZ state: (|00...0⟩ + |11...1⟩)/√2
    
    Maximally entangled n-qubit state.
    """
    dim = 2 ** n_qubits
    amplitudes = np.zeros(dim, dtype=np.complex128)
    amplitudes[0] = 1/np.sqrt(2)
    amplitudes[-1] = 1/np.sqrt(2)
    return QuantumState(n_qubits, amplitudes=amplitudes)


def w_state(n_qubits: int) -> QuantumState:
    """
    Create W state: (|100...⟩ + |010...⟩ + ... + |...001⟩)/√n
    
    Different entanglement structure than GHZ.
    """
    dim = 2 ** n_qubits
    amplitudes = np.zeros(dim, dtype=np.complex128)
    
    for i in range(n_qubits):
        idx = 1 << (n_qubits - 1 - i)
        amplitudes[idx] = 1/np.sqrt(n_qubits)
    
    return QuantumState(n_qubits, amplitudes=amplitudes)


def maximally_mixed(n_qubits: int) -> QuantumState:
    """Create maximally mixed state ρ = I/2^n."""
    dim = 2 ** n_qubits
    rho = np.eye(dim, dtype=np.complex128) / dim
    return QuantumState(n_qubits, density_matrix=rho)


def random_state(n_qubits: int, seed: Optional[int] = None) -> QuantumState:
    """Generate Haar-random pure state."""
    if seed is not None:
        np.random.seed(seed)
    
    dim = 2 ** n_qubits
    # Generate complex Gaussian vector
    real = np.random.randn(dim)
    imag = np.random.randn(dim)
    amplitudes = (real + 1j * imag) / np.sqrt(2)
    
    return QuantumState(n_qubits, amplitudes=amplitudes)
